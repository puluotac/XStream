#include "core/io/ImageFileIo.h"

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

namespace ct
{

cv::Mat readImageFile(const QString& filePath)
{
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
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if (!image.empty() && image.channels() > 1)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    return image;
}

bool writeImageFile(const QString& filePath, const cv::Mat& image, QString* errorMessage)
{
    if (image.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("image is empty");
        }
        return false;
    }

    const QString suffix = QFileInfo(filePath).suffix().isEmpty()
        ? QStringLiteral("tiff")
        : QFileInfo(filePath).suffix();
    const QString extension = QStringLiteral(".") + suffix;
    std::vector<int> params;
    if (extension.compare(QStringLiteral(".tif"), Qt::CaseInsensitive) == 0
        || extension.compare(QStringLiteral(".tiff"), Qt::CaseInsensitive) == 0)
    {
        params = { cv::IMWRITE_TIFF_COMPRESSION, 1 };
    }

    std::vector<uchar> encoded;
    if (!cv::imencode(extension.toStdString(), image, encoded, params))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("cv::imencode failed for extension %1").arg(extension);
        }
        return false;
    }

    QDir().mkpath(QFileInfo(filePath).absolutePath());
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("failed to open output file: %1").arg(file.errorString());
        }
        return false;
    }
    const qint64 written = file.write(reinterpret_cast<const char*>(encoded.data()),
        static_cast<qint64>(encoded.size()));
    if (written != static_cast<qint64>(encoded.size()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("short write: %1 / %2 bytes")
                .arg(written)
                .arg(encoded.size());
        }
        return false;
    }
    return true;
}

} // namespace ct
