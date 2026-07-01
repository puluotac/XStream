#pragma once

#include "hardware/camera/BoundedFrameQueue.h"
#include "hardware/camera/CameraFrame.h"

#include <QObject>
#include <QVector>
#include <QString>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace ct
{

enum class PvcamExposureUnit
{
    Microseconds,
    Milliseconds,
    Seconds
};

struct PvcamCaptureSettings
{
    int frameCount = 10;
    double exposureValue = 10.0;
    PvcamExposureUnit exposureUnit = PvcamExposureUnit::Milliseconds;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 3200;
    int roiHeight = 3200;
    int queueCapacity = 256;
    bool saveRaw = false;
    bool previewOnly = false;
    bool enqueueFrames = true;
    QString outputDir;
    QString filePrefix = QStringLiteral("kinetix_frame_");
    int filePadding = 6;
    QString singleFramePath;
    qint64 previewIntervalMs = 33;
    QVector<int> selectedFrameIndices;
    int rawFrameLimit = 0;
    double frameIntervalMs = 0.0;
    double firstKeepDelayMs = 0.0;
    bool allowFrameIntervalDecimation = false;
};

class PvcamCameraWorker : public QObject
{
    Q_OBJECT

public:
    explicit PvcamCameraWorker(QObject* parent = nullptr);
    ~PvcamCameraWorker() override = default;

public slots:
    void warmOpen();
    void startCapture(const ct::PvcamCaptureSettings& settings);
    void beginPreparedCapture(const ct::PvcamCaptureSettings& settings);
    void capturePreparedFrame(int frameCount = 1);
    void endPreparedCapture();
    void startContinuousCapture(const ct::PvcamCaptureSettings& settings);
    void stopCapture();
    void shutdown();

signals:
    void cameraStateChanged(const QString& state);
    void frameReady(const ct::CameraFrame& frame);
    void captureStatsChanged(qint64 captured, int queued, qint64 consumed, qint64 dropped);
    void logMessage(const QString& text);
    void captureFinished(bool ok, const QString& message);
    void preparedCaptureReady(bool ok, const QString& message);
    void preparedFrameBatchFinished(bool ok, const QString& message);
    void continuousCaptureStarted(bool ok, const QString& message);

private:
    void openCameraIfNeeded();
    void closeCamera();

    std::shared_ptr<BoundedFrameQueue> queue_;
    std::atomic_bool stopRequested_ { false };
    std::atomic_bool capturing_ { false };
    bool preparedActive_ = false;
    bool preparedFrameRunning_ = false;
    PvcamCaptureSettings preparedSettings_;
    qint64 preparedFrameIndex_ = 0;
    qint64 droppedCount_ = 0;

#ifdef CT_HAS_PVCAM
    void cleanupPreparedSequence(bool abortExposure);

    int16_t hcam_ = -1;
    bool pvcamInitialized_ = false;
    bool cameraOpened_ = false;
    QString cameraName_;
    int sensorWidth_ = 0;
    int sensorHeight_ = 0;
    int imageFormat_ = 0;
    int bitDepth_ = 0;
    int preparedWidth_ = 0;
    int preparedHeight_ = 0;
    int preparedCvType_ = 0;
    int preparedBytesPerPixel_ = 0;
    int preparedStrideBytes_ = 0;
    std::int16_t preparedExposureMode_ = 0;
    double preparedExposureMs_ = 0.0;
    std::uint32_t preparedExposureBytes_ = 0;
    std::vector<unsigned char> preparedFrameBuffer_;
#endif
};

} // namespace ct

Q_DECLARE_METATYPE(ct::CameraFrame)
Q_DECLARE_METATYPE(ct::PvcamCaptureSettings)
