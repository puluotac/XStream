#include "core/io/FileProjectionSource.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QThread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <utility>

namespace ct
{

namespace
{

cv::Mat loadImageWithQtPath(const QString& filePath)
{
    cv::Mat image = cv::imread(filePath.toLocal8Bit().constData(), cv::IMREAD_UNCHANGED);
    if (!image.empty())
    {
        return image;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly))
    {
        return {};
    }

    const QByteArray encoded = file.readAll();
    if (encoded.isEmpty())
    {
        return {};
    }

    std::vector<uchar> buffer(encoded.begin(), encoded.end());
    return cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
}

} // namespace

FileProjectionSource::FileProjectionSource(IoConfig config)
    : config_(std::move(config))
{
}

int FileProjectionSource::extractStartIndex(const QString& pattern) const
{
    const QRegularExpression numberRegex(R"((\d+))");
    QRegularExpressionMatchIterator it = numberRegex.globalMatch(pattern);
    int value = 0;
    while (it.hasNext())
    {
        value = it.next().captured(1).toInt();
    }
    return value;
}

QString FileProjectionSource::makeNumberedFileName(const QString& prefix, int index) const
{
    return QString("%1%2.%3")
        .arg(prefix)
        .arg(index, config_.inputPadding, 10, QLatin1Char('0'))
        .arg(config_.inputExt);
}

bool FileProjectionSource::waitForReferenceImage(
    const QString& prefix,
    const QString& label,
    cv::Mat& outImage,
    std::atomic_bool& stopRequested) const
{
    while (!stopRequested.load())
    {
        for (int i = 0; i < 10000 && !stopRequested.load(); ++i)
        {
            const QString fileName = makeNumberedFileName(prefix, i);
            const QString filePath = QDir(config_.watchDir).filePath(fileName);
            outImage = readImageRobust(filePath);
            if (!outImage.empty())
            {
                return true;
            }
        }
        QThread::msleep(200);
    }
    return false;
}

bool FileProjectionSource::waitForSpecificFile(
    const QString& filePattern,
    const QString& label,
    cv::Mat& outImage,
    QString* outPath,
    std::atomic_bool& stopRequested) const
{
    Q_UNUSED(label);

    while (!stopRequested.load())
    {
        QString targetPath = QDir(config_.watchDir).filePath(filePattern);
        if (filePattern.contains("{"))
        {
            targetPath = QDir(config_.watchDir).filePath(QString(filePattern).arg(0));
            if (!QFileInfo::exists(targetPath))
            {
                targetPath = QDir(config_.watchDir).filePath(QString(filePattern).arg(1));
            }
        }

        outImage = readImageRobust(targetPath);
        if (!outImage.empty())
        {
            if (outPath != nullptr)
            {
                *outPath = targetPath;
            }
            return true;
        }

        QThread::msleep(200);
    }
    return false;
}

std::vector<ProjectionFrame> FileProjectionSource::pollNextBatch(
    const std::vector<bool>& processedFlags,
    int startIndex,
    int maxBatchSize,
    std::atomic_bool& stopRequested) const
{
    std::vector<ProjectionFrame> batch;

    for (int logicalIndex = 0; logicalIndex < static_cast<int>(processedFlags.size()) && !stopRequested.load(); ++logicalIndex)
    {
        if (processedFlags[logicalIndex])
        {
            continue;
        }

        const int fileIndex = logicalIndex + startIndex;
        const QString filePath = QDir(config_.watchDir).filePath(makeNumberedFileName(config_.tomoPrefix, fileIndex));
        if (!QFileInfo::exists(filePath))
        {
            continue;
        }

        cv::Mat image = readImageRobust(filePath);
        if (image.empty())
        {
            continue;
        }

        batch.push_back(ProjectionFrame{ logicalIndex, fileIndex, filePath, image });
        if (static_cast<int>(batch.size()) >= maxBatchSize)
        {
            break;
        }
    }

    return batch;
}

cv::Mat FileProjectionSource::readImageRobust(const QString& filePath) const
{
    for (int attempt = 0; attempt < 5; ++attempt)
    {
        QFileInfo info(filePath);
        if (info.exists() && info.size() > 0)
        {
            cv::Mat image = loadImageWithQtPath(filePath);
            if (!image.empty())
            {
                if (image.channels() > 1)
                {
                    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
                }
                return image;
            }
        }
        QThread::msleep(10);
    }
    return {};
}

} // namespace ct
