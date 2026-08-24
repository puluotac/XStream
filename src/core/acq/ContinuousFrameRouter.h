#pragma once

#include "core/acq/ContinuousFrameSink.h"
#include "core/acq/CtTiffWriter.h"

#include <QObject>
#include <QVector>

#include <condition_variable>
#include <mutex>
#include <thread>

namespace ct
{

struct ContinuousFrameRouteSettings
{
    QString savePath;
    QString tomoPrefix = QStringLiteral("Tomo_");
    QString timeSeriesPrefix = QStringLiteral("Frame_");
    QString alignment180Path;
    QString extension = QStringLiteral("tiff");
    int padding = 4;
    int projectionCount = 0;
    bool flyScan = false;
    bool autoCor = false;
    bool previewEnabled = true;
    int previewIntervalMs = 50;
};

struct ContinuousFrameMetadata
{
    qint64 rawFrameIndex = 0;
    qint64 logicalIndex = -1;
    qint64 deviceFrameNumber = 0;
    qint64 bofTimestampNs = 0;
    qint64 eofTimestampNs = 0;
    double elapsedMs = 0.0;
    qint64 timestampMs = 0;
    int fileIndex = -1;
    bool cor180 = false;
    QString filePath;
    QString timingSource;
};

class ContinuousFrameRouter final : public QObject, public ContinuousFrameSink
{
    Q_OBJECT

public:
    explicit ContinuousFrameRouter(QObject* parent = nullptr);
    ~ContinuousFrameRouter() override;

    void configure(const ContinuousFrameRouteSettings& settings, CtTiffWriter* writer);
    void start();
    void stopPreview();

    bool submitContinuousFrame(const CameraFrame& frame, QString* errorMessage) override;
    void continuousCaptureFinished(bool ok, const QString& message) override;

signals:
    void previewFrameReady(const ct::CameraFrame& frame);
    void frameBatchFinished(const QVector<ct::ContinuousFrameMetadata>& frames, bool ok, const QString& message);

private:
    QString numberedPath(const QString& prefix, int index) const;
    CameraFrame makePreviewFrame(const CameraFrame& source) const;
    void previewLoop();

    ContinuousFrameRouteSettings settings_;
    CtTiffWriter* writer_ = nullptr;
    QVector<ContinuousFrameMetadata> routedFrames_;
    std::mutex previewMutex_;
    std::condition_variable previewCondition_;
    CameraFrame latestPreviewFrame_;
    bool hasLatestPreview_ = false;
    bool previewStopRequested_ = false;
    std::thread previewThread_;
    bool started_ = false;
};

} // namespace ct

Q_DECLARE_METATYPE(ct::ContinuousFrameMetadata)
Q_DECLARE_METATYPE(QVector<ct::ContinuousFrameMetadata>)
