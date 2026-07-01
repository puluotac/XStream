#pragma once

#include "core/acq/CtAcquisitionTypes.h"
#include "hardware/camera/CameraFrame.h"

#include <QObject>
#include <QJsonObject>
#include <QString>
#include <QVector>

#include <atomic>

namespace ct
{

struct CtReplayAcquisitionSettings
{
    QString sourceProjectionDir;
    QString sourceJobPath;
    QString targetProjectionDir;
    QString targetOutputDir;
    QString tomoPrefix;
    QString flatPrefix;
    QString inputExt = QStringLiteral("tiff");
    QString filePattern0;
    QString filePattern180;
    int inputPadding = 4;
    int projectionCount = 0;
    int imageWidth = 0;
    int imageHeight = 0;
    int frameIntervalMs = 0;
    bool autoCor = false;
    bool corReferenceCaptured = false;
    QJsonObject manifestTemplate;
    QJsonObject sourceJob;
    QVector<CtFlatGroup> flatGroups;
};

class CtReplayAcquisitionWorker : public QObject
{
    Q_OBJECT

public:
    explicit CtReplayAcquisitionWorker(QObject* parent = nullptr);

public slots:
    void startReplay(const ct::CtReplayAcquisitionSettings& settings);
    void stopReplay();

signals:
    void frameReady(const ct::CameraFrame& frame, const QString& filePath, bool isFlat, bool isCor180);
    void logMessage(const QString& text);
    void replayFinished(bool ok, const QString& message);

private:
    std::atomic_bool stopRequested_ { false };
};

} // namespace ct

Q_DECLARE_METATYPE(ct::CtReplayAcquisitionSettings)
