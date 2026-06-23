#include "hardware/camera/PvcamCameraWorker.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QMetaType>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef CT_HAS_PVCAM
#include <master.h>
#include <pvcam.h>
#endif

namespace ct
{

namespace
{

#ifdef CT_HAS_PVCAM

constexpr qint64 kPreviewSignalMinIntervalMs = 33;

QString pvcamError()
{
    char message[ERROR_MSG_LEN] = {};
    const int16 code = pl_error_code();
    pl_error_message(code, message);
    return QStringLiteral("PVCAM error %1: %2").arg(code).arg(QString::fromLatin1(message));
}

template <typename T>
bool getParam(int16 hcam, uns32 id, int16 attr, T& value)
{
    return PV_OK == pl_get_param(hcam, id, attr, &value);
}

bool enumContains(int16 hcam, uns32 paramId, int32 wanted)
{
    uns32 count = 0;
    if (PV_OK != pl_get_param(hcam, paramId, ATTR_COUNT, &count))
    {
        return false;
    }

    for (uns32 index = 0; index < count; ++index)
    {
        int32 value = 0;
        uns32 length = 0;
        if (PV_OK != pl_enum_str_length(hcam, paramId, index, &length) || length == 0)
        {
            continue;
        }

        std::vector<char> text(length + 1, '\0');
        if (PV_OK == pl_get_enum_param(hcam, paramId, index, &value, text.data(), length)
            && value == wanted)
        {
            return true;
        }
    }
    return false;
}

int16 selectInternalExposureMode(int16 hcam)
{
    if (enumContains(hcam, PARAM_EXPOSURE_MODE, EXT_TRIG_INTERNAL))
    {
        int16 exposeOut = EXPOSE_OUT_FIRST_ROW;
        uns32 exposeOutCount = 0;
        if (PV_OK == pl_get_param(hcam, PARAM_EXPOSE_OUT_MODE, ATTR_COUNT, &exposeOutCount)
            && exposeOutCount > 0)
        {
            int32 value = 0;
            uns32 length = 0;
            if (PV_OK == pl_enum_str_length(hcam, PARAM_EXPOSE_OUT_MODE, 0, &length) && length > 0)
            {
                std::vector<char> text(length + 1, '\0');
                if (PV_OK == pl_get_enum_param(hcam, PARAM_EXPOSE_OUT_MODE, 0, &value, text.data(), length))
                {
                    exposeOut = static_cast<int16>(value);
                }
            }
        }
        return static_cast<int16>(EXT_TRIG_INTERNAL | exposeOut);
    }

    if (enumContains(hcam, PARAM_EXPOSURE_MODE, TIMED_MODE))
    {
        return TIMED_MODE;
    }

    throw std::runtime_error("Camera does not report an internal timed exposure mode.");
}

rgn_type makeCenteredRoi(uns16 sensorWidth, uns16 sensorHeight, const PvcamCaptureSettings& settings)
{
    const int roiWidth = settings.fullFrame ? sensorWidth : std::min<int>(settings.roiWidth, sensorWidth);
    const int roiHeight = settings.fullFrame ? sensorHeight : std::min<int>(settings.roiHeight, sensorHeight);
    const int x0 = (static_cast<int>(sensorWidth) - roiWidth) / 2;
    const int y0 = (static_cast<int>(sensorHeight) - roiHeight) / 2;

    rgn_type roi {};
    roi.s1 = static_cast<uns16>(x0);
    roi.s2 = static_cast<uns16>(x0 + roiWidth - 1);
    roi.sbin = 1;
    roi.p1 = static_cast<uns16>(y0);
    roi.p2 = static_cast<uns16>(y0 + roiHeight - 1);
    roi.pbin = 1;
    return roi;
}

int roiWidth(const rgn_type& roi)
{
    return (roi.s2 - roi.s1 + 1) / roi.sbin;
}

int roiHeight(const rgn_type& roi)
{
    return (roi.p2 - roi.p1 + 1) / roi.pbin;
}

int cvTypeForImageFormat(int32 imageFormat)
{
    switch (imageFormat)
    {
    case PL_IMAGE_FORMAT_MONO8:
    case PL_IMAGE_FORMAT_BAYER8:
        return CV_8UC1;
    case PL_IMAGE_FORMAT_MONO16:
    case PL_IMAGE_FORMAT_BAYER16:
        return CV_16UC1;
    case PL_IMAGE_FORMAT_MONO32:
    case PL_IMAGE_FORMAT_BAYER32:
        return CV_32SC1;
    default:
        throw std::runtime_error("Unsupported PVCAM image format: " + std::to_string(imageFormat));
    }
}

int bytesPerPixelForCvType(int cvType)
{
    return static_cast<int>(CV_ELEM_SIZE(cvType));
}

bool waitForReadout(int16 hcam, uns32 timeoutMs, std::atomic_bool& stopRequested, uns32& bytesArrived)
{
    const qint64 deadline = QDateTime::currentMSecsSinceEpoch() + timeoutMs;
    int16 status = 0;
    bytesArrived = 0;

    while (QDateTime::currentMSecsSinceEpoch() < deadline)
    {
        if (stopRequested.load())
        {
            return false;
        }

        if (PV_OK != pl_exp_check_status(hcam, &status, &bytesArrived))
        {
            return false;
        }

        if (status == READOUT_COMPLETE || status == FRAME_AVAILABLE)
        {
            return true;
        }

        if (status != EXPOSURE_IN_PROGRESS && status != READOUT_IN_PROGRESS)
        {
            return false;
        }
    }
    return false;
}

QString makeFramePath(const QString& outputDir, qint64 frameIndex)
{
    std::ostringstream name;
    name << "kinetix_frame_" << std::setw(6) << std::setfill('0') << frameIndex << ".tiff";
    return QDir(outputDir).filePath(QString::fromStdString(name.str()));
}

#endif

} // namespace

PvcamCameraWorker::PvcamCameraWorker(QObject* parent)
    : QObject(parent)
    , queue_(std::make_shared<BoundedFrameQueue>(256))
{
    qRegisterMetaType<ct::CameraFrame>("ct::CameraFrame");
    qRegisterMetaType<ct::PvcamCaptureSettings>("ct::PvcamCaptureSettings");
}

void PvcamCameraWorker::startCapture(const PvcamCaptureSettings& rawSettings)
{
    if (capturing_.exchange(true))
    {
        emit logMessage(QStringLiteral("Kinetix capture is already running."));
        return;
    }

    PvcamCaptureSettings settings = rawSettings;
    settings.frameCount = std::max(1, settings.frameCount);
    settings.exposureMs = std::max(1, settings.exposureMs);
    settings.roiWidth = std::max(1, settings.roiWidth);
    settings.roiHeight = std::max(1, settings.roiHeight);
    settings.queueCapacity = std::max(1, settings.queueCapacity);

    stopRequested_.store(false);
    droppedCount_ = 0;
    queue_->setCapacity(settings.queueCapacity);
    emit captureStatsChanged(0, 0, 0, 0);

#ifndef CT_HAS_PVCAM
    capturing_.store(false);
    emit cameraStateChanged(QStringLiteral("PVCAM unavailable"));
    emit captureFinished(false, QStringLiteral("PVCAM SDK was not found at build time."));
    return;
#else
    int16 hcam = -1;
    bool pvcamInitialized = false;
    bool cameraOpened = false;
    bool sequenceInitialized = false;

    try
    {
        emit cameraStateChanged(QStringLiteral("Opening"));
        if (PV_OK != pl_pvcam_init())
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        pvcamInitialized = true;

        int16 total = 0;
        if (PV_OK != pl_cam_get_total(&total) || total <= 0)
        {
            throw std::runtime_error("No PVCAM cameras found.");
        }

        char cameraName[CAM_NAME_LEN] = {};
        if (PV_OK != pl_cam_get_name(0, cameraName))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        if (PV_OK != pl_cam_open(cameraName, &hcam, OPEN_EXCLUSIVE))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        cameraOpened = true;
        emit logMessage(QStringLiteral("Kinetix opened: %1").arg(QString::fromLatin1(cameraName)));

        uns16 sensorWidth = 0;
        uns16 sensorHeight = 0;
        if (!getParam(hcam, PARAM_SER_SIZE, ATTR_CURRENT, sensorWidth)
            || !getParam(hcam, PARAM_PAR_SIZE, ATTR_CURRENT, sensorHeight))
        {
            throw std::runtime_error("Failed to read PVCAM sensor size.");
        }

        int32 imageFormat = 0;
        if (!getParam(hcam, PARAM_IMAGE_FORMAT, ATTR_CURRENT, imageFormat))
        {
            throw std::runtime_error("Failed to read PVCAM image format.");
        }

        int16 bitDepth = 0;
        getParam(hcam, PARAM_BIT_DEPTH, ATTR_CURRENT, bitDepth);

        const int cvType = cvTypeForImageFormat(imageFormat);
        const int bytesPerPixel = bytesPerPixelForCvType(cvType);
        const rgn_type roi = makeCenteredRoi(sensorWidth, sensorHeight, settings);
        const int width = roiWidth(roi);
        const int height = roiHeight(roi);
        const int strideBytes = width * bytesPerPixel;

        const int16 expMode = selectInternalExposureMode(hcam);
        uns32 exposureBytes = 0;
        rgn_type mutableRoi = roi;
        if (PV_OK != pl_exp_setup_seq(hcam, 1, 1, &mutableRoi, expMode,
                static_cast<uns32>(settings.exposureMs), &exposureBytes))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        sequenceInitialized = true;

        const uns32 expectedBytes = static_cast<uns32>(height * strideBytes);
        if (exposureBytes < expectedBytes)
        {
            throw std::runtime_error("PVCAM returned fewer bytes than the selected ROI requires.");
        }

        emit logMessage(QStringLiteral("Kinetix ROI %1x%2, bitDepth=%3, cvType=%4, stride=%5, frameBytes=%6.")
            .arg(width)
            .arg(height)
            .arg(bitDepth)
            .arg(cvType)
            .arg(strideBytes)
            .arg(exposureBytes));

        if (settings.saveRaw)
        {
            QDir().mkpath(settings.outputDir);
            if (!QFileInfo(settings.outputDir).isDir())
            {
                throw std::runtime_error("Output directory is unavailable.");
            }
        }

        std::vector<unsigned char> frameBuffer(exposureBytes);
        emit cameraStateChanged(settings.previewOnly ? QStringLiteral("Live") : QStringLiteral("Capturing"));
        qint64 lastPreviewEmitMs = -kPreviewSignalMinIntervalMs;

        for (int frameIndex = 0; (settings.previewOnly || frameIndex < settings.frameCount) && !stopRequested_.load(); ++frameIndex)
        {
            if (PV_OK != pl_exp_start_seq(hcam, frameBuffer.data()))
            {
                throw std::runtime_error(pvcamError().toStdString());
            }

            uns32 bytesArrived = 0;
            const uns32 timeoutMs = std::max<uns32>(5000, static_cast<uns32>(settings.exposureMs + 5000));
            if (!waitForReadout(hcam, timeoutMs, stopRequested_, bytesArrived))
            {
                pl_exp_abort(hcam, CCS_HALT);
                if (stopRequested_.load())
                {
                    break;
                }
                throw std::runtime_error("Timed out waiting for Kinetix frame.");
            }

            cv::Mat wrapped(height, width, cvType, frameBuffer.data(), static_cast<size_t>(strideBytes));
            CameraFrame frame;
            frame.frameIndex = frameIndex;
            frame.timestampMs = QDateTime::currentMSecsSinceEpoch();
            frame.width = width;
            frame.height = height;
            frame.bitDepth = bitDepth;
            frame.bytesPerPixel = bytesPerPixel;
            frame.strideBytes = strideBytes;
            frame.cvType = cvType;
            frame.image = wrapped.clone();

            if (PV_OK != pl_exp_finish_seq(hcam, frameBuffer.data(), 0))
            {
                emit logMessage(QStringLiteral("pl_exp_finish_seq warning: %1").arg(pvcamError()));
            }

            if (!settings.previewOnly && !queue_->push(frame))
            {
                ++droppedCount_;
                throw std::runtime_error("Camera frame queue is full; capture stopped to avoid frame loss.");
            }

            if (!settings.previewOnly && settings.saveRaw)
            {
                const QString framePath = makeFramePath(settings.outputDir, frame.frameIndex);
                if (!cv::imwrite(framePath.toStdString(), frame.image))
                {
                    throw std::runtime_error("Failed to save camera frame.");
                }
            }

            const bool isLastFiniteCaptureFrame = !settings.previewOnly && (frameIndex + 1) >= settings.frameCount;
            const bool shouldEmitPreview =
                frameIndex == 0
                || isLastFiniteCaptureFrame
                || frame.timestampMs - lastPreviewEmitMs >= kPreviewSignalMinIntervalMs;
            if (shouldEmitPreview)
            {
                lastPreviewEmitMs = frame.timestampMs;
                emit frameReady(frame);
            }
            emit captureStatsChanged(frame.frameIndex + 1, queue_->size(), queue_->consumedCount(), droppedCount_);
        }

        if (sequenceInitialized)
        {
            pl_exp_uninit_seq();
            sequenceInitialized = false;
        }
        if (cameraOpened)
        {
            pl_cam_close(hcam);
            cameraOpened = false;
        }
        if (pvcamInitialized)
        {
            pl_pvcam_uninit();
            pvcamInitialized = false;
        }

        capturing_.store(false);
        emit cameraStateChanged(QStringLiteral("Idle"));
        emit captureFinished(!stopRequested_.load(), stopRequested_.load() ? QStringLiteral("Capture stopped.") : QStringLiteral("Capture completed."));
    }
    catch (const std::exception& ex)
    {
        if (sequenceInitialized)
        {
            pl_exp_abort(hcam, CCS_HALT);
            pl_exp_uninit_seq();
        }
        if (cameraOpened)
        {
            pl_cam_close(hcam);
        }
        if (pvcamInitialized)
        {
            pl_pvcam_uninit();
        }

        capturing_.store(false);
        emit cameraStateChanged(QStringLiteral("Error"));
        emit captureFinished(false, QString::fromLocal8Bit(ex.what()));
    }
#endif
}

void PvcamCameraWorker::stopCapture()
{
    stopRequested_.store(true);
}

void PvcamCameraWorker::shutdown()
{
    stopCapture();
}

} // namespace ct
