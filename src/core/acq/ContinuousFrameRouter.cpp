#include "core/acq/ContinuousFrameRouter.h"

#include <algorithm>
#include <chrono>

#include <QDir>

#include <opencv2/imgproc.hpp>

namespace ct
{

ContinuousFrameRouter::ContinuousFrameRouter(QObject* parent)
    : QObject(parent)
{
}

ContinuousFrameRouter::~ContinuousFrameRouter()
{
    stopPreview();
}

void ContinuousFrameRouter::configure(const ContinuousFrameRouteSettings& settings, CtTiffWriter* writer)
{
    stopPreview();
    settings_ = settings;
    settings_.padding = std::clamp(settings_.padding, 1, 12);
    settings_.previewIntervalMs = std::clamp(settings_.previewIntervalMs, 16, 1000);
    writer_ = writer;
    routedFrames_.clear();
    latestPreviewFrame_ = {};
    hasLatestPreview_ = false;
    started_ = false;
}

void ContinuousFrameRouter::start()
{
    stopPreview();
    {
        std::lock_guard<std::mutex> lock(previewMutex_);
        previewStopRequested_ = false;
        hasLatestPreview_ = false;
    }
    routedFrames_.clear();
    started_ = true;
    if (settings_.previewEnabled)
    {
        previewThread_ = std::thread([this]() { previewLoop(); });
    }
}

void ContinuousFrameRouter::stopPreview()
{
    {
        std::lock_guard<std::mutex> lock(previewMutex_);
        previewStopRequested_ = true;
        hasLatestPreview_ = false;
        latestPreviewFrame_ = {};
    }
    previewCondition_.notify_all();
    if (previewThread_.joinable())
    {
        previewThread_.join();
    }
    started_ = false;
}

QString ContinuousFrameRouter::numberedPath(const QString& prefix, int index) const
{
    return QDir(settings_.savePath).filePath(QStringLiteral("%1%2.%3")
        .arg(prefix)
        .arg(std::max(0, index), settings_.padding, 10, QLatin1Char('0'))
        .arg(settings_.extension));
}

bool ContinuousFrameRouter::submitContinuousFrame(const CameraFrame& frame, QString* errorMessage)
{
    if (!started_ || writer_ == nullptr)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Continuous frame router is not ready.");
        }
        return false;
    }

    const int logicalIndex = static_cast<int>(frame.logicalIndex >= 0
        ? frame.logicalIndex
        : routedFrames_.size());
    const bool cor180 = settings_.flyScan && settings_.autoCor
        && logicalIndex == settings_.projectionCount;
    const int fileIndex = cor180 ? -1 : logicalIndex;
    const QString filePath = cor180
        ? settings_.alignment180Path
        : numberedPath(settings_.flyScan ? settings_.tomoPrefix : settings_.timeSeriesPrefix, fileIndex);

    CtFrameSaveItem item;
    item.filePath = filePath;
    item.bufferLease = frame.bufferLease;
    item.width = frame.width;
    item.height = frame.height;
    item.cvType = frame.cvType;
    item.strideBytes = frame.strideBytes;
    if (!item.bufferLease && !frame.image.empty())
    {
        item.ownedImage = std::make_shared<cv::Mat>(frame.image);
    }

    QString enqueueError;
    if (!writer_->tryEnqueue(item, &enqueueError))
    {
        if (errorMessage)
        {
            *errorMessage = enqueueError;
        }
        return false;
    }

    ContinuousFrameMetadata metadata;
    metadata.rawFrameIndex = frame.frameIndex;
    metadata.logicalIndex = frame.logicalIndex;
    metadata.deviceFrameNumber = frame.deviceFrameNumber;
    metadata.bofTimestampNs = frame.bofTimestampNs;
    metadata.eofTimestampNs = frame.eofTimestampNs;
    metadata.elapsedMs = frame.elapsedMs;
    metadata.timestampMs = frame.timestampMs;
    metadata.fileIndex = fileIndex;
    metadata.cor180 = cor180;
    metadata.filePath = filePath;
    metadata.timingSource = frame.timingSource;
    routedFrames_.append(metadata);

    if (settings_.previewEnabled)
    {
        {
            std::lock_guard<std::mutex> lock(previewMutex_);
            latestPreviewFrame_ = frame;
            hasLatestPreview_ = true;
        }
        previewCondition_.notify_one();
    }
    return true;
}

void ContinuousFrameRouter::continuousCaptureFinished(bool ok, const QString& message)
{
    if (!started_)
    {
        return;
    }
    stopPreview();
    emit frameBatchFinished(routedFrames_, ok, message);
}

CameraFrame ContinuousFrameRouter::makePreviewFrame(const CameraFrame& source) const
{
    CameraFrame result;
    result.frameIndex = source.frameIndex;
    result.logicalIndex = source.logicalIndex;
    result.deviceFrameNumber = source.deviceFrameNumber;
    result.bofTimestampNs = source.bofTimestampNs;
    result.eofTimestampNs = source.eofTimestampNs;
    result.elapsedMs = source.elapsedMs;
    result.timestampMs = source.timestampMs;
    result.width = source.width;
    result.height = source.height;
    result.bitDepth = source.bitDepth;
    result.bytesPerPixel = source.bytesPerPixel;
    result.timingSource = source.timingSource;

    cv::Mat sourceImage;
    if (!source.image.empty())
    {
        sourceImage = source.image;
    }
    else if (source.bufferLease && source.bufferLease->data() != nullptr
        && source.width > 0 && source.height > 0 && source.strideBytes > 0)
    {
        sourceImage = cv::Mat(source.height, source.width, source.cvType,
            source.bufferLease->data(), static_cast<size_t>(source.strideBytes));
    }
    if (sourceImage.empty())
    {
        return result;
    }

    constexpr int maxPreviewSide = 1024;
    const int maxSide = std::max(sourceImage.cols, sourceImage.rows);
    if (maxSide > maxPreviewSide)
    {
        const double scale = static_cast<double>(maxPreviewSide) / static_cast<double>(maxSide);
        cv::resize(sourceImage, result.image, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    else
    {
        result.image = sourceImage.clone();
    }
    result.strideBytes = static_cast<int>(result.image.step);
    result.cvType = result.image.type();
    return result;
}

void ContinuousFrameRouter::previewLoop()
{
    auto nextEmit = std::chrono::steady_clock::now();
    for (;;)
    {
        CameraFrame source;
        {
            std::unique_lock<std::mutex> lock(previewMutex_);
            for (;;)
            {
                if (previewStopRequested_)
                {
                    return;
                }

                const auto now = std::chrono::steady_clock::now();
                if (hasLatestPreview_ && now >= nextEmit)
                {
                    source = latestPreviewFrame_;
                    hasLatestPreview_ = false;
                    nextEmit = now + std::chrono::milliseconds(settings_.previewIntervalMs);
                    break;
                }

                // A new camera frame only replaces the latest source. It must
                // not wake the preview into rendering faster than the requested
                // UI cadence, otherwise preview work can again steal capture CPU.
                const auto wakeAt = hasLatestPreview_
                    ? nextEmit
                    : now + std::chrono::milliseconds(settings_.previewIntervalMs);
                previewCondition_.wait_until(lock, wakeAt);
            }
        }

        CameraFrame preview = makePreviewFrame(source);
        if (!preview.image.empty())
        {
            emit previewFrameReady(preview);
        }
    }
}

} // namespace ct
