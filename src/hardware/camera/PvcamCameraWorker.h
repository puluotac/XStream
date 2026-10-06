#pragma once

#include "core/acq/ContinuousTiming.h"
#include "core/acq/ContinuousFrameSink.h"
#include "core/acq/FrameBufferPool.h"
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
    int rawFrameStride = 1;
    double baselineRawFramePeriodMs = 0.0;
    // Bounded in-flight frame storage.  CT routing persists frames through
    // the writer; this is deliberately not sized to the projection count.
    int framePoolCapacity = 128;
    int ringFrameCount = 192;
    bool waitForFrameGate = false;
};

struct PvcamTimingCalibrationSettings
{
    double exposureValue = 10.0;
    PvcamExposureUnit exposureUnit = PvcamExposureUnit::Milliseconds;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 3200;
    int roiHeight = 3200;
    int warmupFrames = 3;
    int sampleFrames = 10;
    int ringFrameCount = 64;
};

struct PvcamTimingCalibrationResult
{
    bool ok = false;
    QString message;
    QString timingSource;
    double exposureMs = 0.0;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 0;
    int roiHeight = 0;
    int bitDepth = 0;
    int metadataVersion = 0;
    QString cameraName;
    QString cameraPartNumber;
    QString cameraSerial;
    QString readoutMode;
    QString readoutPort;
    QString readoutSpeed;
    QString gain;
    QString triggerMode;
    int binningX = 1;
    int binningY = 1;
    TimingAnalysis analysis;
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
    void calibrateContinuousTiming(const ct::PvcamTimingCalibrationSettings& settings);
    void armContinuousFrameGate();
    void startContinuousCapture(const ct::PvcamCaptureSettings& settings);
    void stopCapture();
    void shutdown();

    void setContinuousFrameSink(ContinuousFrameSink* sink);

signals:
    void cameraStateChanged(const QString& state);
    void frameReady(const ct::CameraFrame& frame);
    void captureStatsChanged(qint64 captured, int queued, qint64 consumed, qint64 dropped);
    void logMessage(const QString& text);
    void captureFinished(bool ok, const QString& message);
    void preparedCaptureReady(bool ok, const QString& message);
    void preparedFrameBatchFinished(bool ok, const QString& message);
    void continuousCaptureStarted(bool ok, const QString& message);
    void continuousIntervalStrategyResolved(const QString& strategy, const QString& reason);
    void timingCalibrationFinished(const ct::PvcamTimingCalibrationResult& result);
    void continuousStatsChanged(qint64 rawFrames, qint64 keptFrames, qint64 discardedFrames,
        qint64 deviceDroppedFrames, int poolInUse, int poolPeak);
    void continuousRingStatsChanged(int estimatedUnread, int ringCapacity, int peakEstimatedUnread);
    void continuousTimingStatsChanged(const ct::ContinuousTimingStats& stats);

private:
    void openCameraIfNeeded();
    void closeCamera();
    bool ensureFramePool(qsizetype frameBytes, int capacity, QString* errorMessage);

    std::shared_ptr<BoundedFrameQueue> queue_;
    std::atomic_bool stopRequested_ { false };
    std::atomic_bool capturing_ { false };
    bool preparedActive_ = false;
    bool preparedFrameRunning_ = false;
    PvcamCaptureSettings preparedSettings_;
    qint64 preparedFrameIndex_ = 0;
    qint64 droppedCount_ = 0;
    FrameBufferPool framePool_;
    std::atomic_bool continuousGateArmed_ { false };
    std::atomic<qint64> continuousGateArmSteadyNs_ { 0 };
    ContinuousFrameSink* continuousFrameSink_ = nullptr;

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
Q_DECLARE_METATYPE(ct::PvcamTimingCalibrationSettings)
Q_DECLARE_METATYPE(ct::PvcamTimingCalibrationResult)
Q_DECLARE_METATYPE(ct::ContinuousTimingStats)
