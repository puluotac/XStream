#pragma once

#include "hardware/camera/BoundedFrameQueue.h"
#include "hardware/camera/CameraFrame.h"

#include <QObject>
#include <QString>

#include <atomic>
#include <memory>

namespace ct
{

struct PvcamCaptureSettings
{
    int frameCount = 10;
    int exposureMs = 10;
    int roiWidth = 512;
    int roiHeight = 512;
    bool fullFrame = false;
    int queueCapacity = 256;
    bool saveRaw = false;
    bool previewOnly = false;
    QString outputDir;
};

class PvcamCameraWorker : public QObject
{
    Q_OBJECT

public:
    explicit PvcamCameraWorker(QObject* parent = nullptr);
    ~PvcamCameraWorker() override = default;

public slots:
    void startCapture(const ct::PvcamCaptureSettings& settings);
    void stopCapture();
    void shutdown();

signals:
    void cameraStateChanged(const QString& state);
    void frameReady(const ct::CameraFrame& frame);
    void captureStatsChanged(qint64 captured, int queued, qint64 consumed, qint64 dropped);
    void logMessage(const QString& text);
    void captureFinished(bool ok, const QString& message);

private:
    std::shared_ptr<BoundedFrameQueue> queue_;
    std::atomic_bool stopRequested_ { false };
    std::atomic_bool capturing_ { false };
    qint64 droppedCount_ = 0;
};

} // namespace ct

Q_DECLARE_METATYPE(ct::CameraFrame)
Q_DECLARE_METATYPE(ct::PvcamCaptureSettings)
