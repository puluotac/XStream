#include "core/acq/CtReplayAcquisitionWorker.h"

#include "core/io/ImageFileIo.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSet>
#include <QThread>

#include <algorithm>

namespace ct
{

namespace
{

QString numberedFileName(const QString& prefix, int padding, int index, const QString& ext)
{
    QString normalizedExt = ext;
    if (normalizedExt.startsWith(QLatin1Char('.')))
    {
        normalizedExt.remove(0, 1);
    }
    return QStringLiteral("%1%2.%3")
        .arg(prefix)
        .arg(std::max(0, index), std::clamp(padding, 1, 12), 10, QLatin1Char('0'))
        .arg(normalizedExt);
}

int tomoStartIndex(const CtReplayAcquisitionSettings& settings)
{
    if (!settings.filePattern0.isEmpty())
    {
        const QRegularExpression re(QStringLiteral("(\\d+)(?=\\.[^.]+$)"));
        const QRegularExpressionMatch match = re.match(settings.filePattern0);
        if (match.hasMatch())
        {
            bool ok = false;
            const int value = match.captured(1).toInt(&ok);
            if (ok)
            {
                return value;
            }
        }
    }
    const QString zeroPath = QDir(settings.sourceProjectionDir).filePath(numberedFileName(settings.tomoPrefix,
        settings.inputPadding, 0, settings.inputExt));
    return QFileInfo::exists(zeroPath) ? 0 : 1;
}

bool appendFrame(QVector<QPair<QString, QString>>& frames, QSet<QString>& targets,
    const QString& sourcePath, const QString& targetPath, QString* error)
{
    if (!QFileInfo::exists(sourcePath))
    {
        if (error)
        {
            *error = QStringLiteral("missing source frame: %1").arg(sourcePath);
        }
        return false;
    }
    const QString normalizedTarget = QDir::fromNativeSeparators(QFileInfo(targetPath).absoluteFilePath());
    if (targets.contains(normalizedTarget))
    {
        return true;
    }
    targets.insert(normalizedTarget);
    frames.append({ sourcePath, targetPath });
    return true;
}

} // namespace

CtReplayAcquisitionWorker::CtReplayAcquisitionWorker(QObject* parent)
    : QObject(parent)
{
}

void CtReplayAcquisitionWorker::startReplay(const CtReplayAcquisitionSettings& settings)
{
    stopRequested_ = false;
    QVector<QPair<QString, QString>> frames;
    QSet<QString> targetPaths;
    QString error;

    if (settings.autoCor && settings.corReferenceCaptured && !settings.filePattern180.isEmpty())
    {
        if (!appendFrame(frames, targetPaths,
                QDir(settings.sourceProjectionDir).filePath(settings.filePattern180),
                QDir(settings.targetProjectionDir).filePath(settings.filePattern180),
                &error))
        {
            emit replayFinished(false, error);
            return;
        }
    }

    for (const CtFlatGroup& group : settings.flatGroups)
    {
        for (int i = 0; i < group.flatCount; ++i)
        {
            const QString name = numberedFileName(settings.flatPrefix, settings.inputPadding,
                group.firstFlatIndex + i, settings.inputExt);
            if (!appendFrame(frames, targetPaths,
                    QDir(settings.sourceProjectionDir).filePath(name),
                    QDir(settings.targetProjectionDir).filePath(name),
                    &error))
            {
                emit replayFinished(false, error);
                return;
            }
        }
    }

    const int startIndex = tomoStartIndex(settings);
    for (int i = 0; i < settings.projectionCount; ++i)
    {
        const QString name = numberedFileName(settings.tomoPrefix, settings.inputPadding,
            startIndex + i, settings.inputExt);
        if (!appendFrame(frames, targetPaths,
                QDir(settings.sourceProjectionDir).filePath(name),
                QDir(settings.targetProjectionDir).filePath(name),
                &error))
        {
            emit replayFinished(false, error);
            return;
        }
    }

    emit logMessage(QStringLiteral("Internal CT replay started: %1 frames from %2 to %3.")
        .arg(frames.size())
        .arg(settings.sourceProjectionDir)
        .arg(settings.targetProjectionDir));
    QElapsedTimer timer;
    timer.start();
    int emitted = 0;
    for (const auto& pair : frames)
    {
        if (stopRequested_)
        {
            emit replayFinished(false, QStringLiteral("Internal CT replay stopped."));
            return;
        }
        cv::Mat image = readImageFile(pair.first);
        if (image.empty())
        {
            emit replayFinished(false, QStringLiteral("failed to read source frame: %1").arg(pair.first));
            return;
        }
        if (image.type() != CV_16U)
        {
            emit replayFinished(false, QStringLiteral("source frame is not uint16: %1").arg(pair.first));
            return;
        }

        CameraFrame frame;
        frame.frameIndex = emitted;
        frame.timestampMs = timer.elapsed();
        frame.width = image.cols;
        frame.height = image.rows;
        frame.bitDepth = 16;
        frame.bytesPerPixel = static_cast<int>(image.elemSize());
        frame.strideBytes = static_cast<int>(image.step);
        frame.cvType = image.type();
        frame.image = image;

        const QString fileName = QFileInfo(pair.second).fileName();
        const bool isCor180 = fileName == settings.filePattern180;
        const bool isFlat = !isCor180 && fileName.startsWith(settings.flatPrefix);
        emit frameReady(frame, pair.second, isFlat, isCor180);
        ++emitted;
        if ((emitted % 50) == 0 || emitted == frames.size())
        {
            emit logMessage(QStringLiteral("Internal CT replay emitted %1/%2 frames.").arg(emitted).arg(frames.size()));
        }
        if (settings.frameIntervalMs > 0)
        {
            QThread::msleep(static_cast<unsigned long>(settings.frameIntervalMs));
        }
    }

    emit replayFinished(true, QStringLiteral("Internal CT replay emitted %1 frames.").arg(emitted));
}

void CtReplayAcquisitionWorker::stopReplay()
{
    stopRequested_ = true;
}

} // namespace ct
