#include "core/io/ImageFileIo.h"

#include <QByteArray>
#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <limits>
#include <vector>

namespace ct
{

namespace
{

constexpr quint16 kTiffTypeShort = 3;
constexpr quint16 kTiffTypeLong = 4;
constexpr quint16 kTiffTagImageWidth = 256;
constexpr quint16 kTiffTagImageLength = 257;
constexpr quint16 kTiffTagBitsPerSample = 258;
constexpr quint16 kTiffTagCompression = 259;
constexpr quint16 kTiffTagPhotometricInterpretation = 262;
constexpr quint16 kTiffTagStripOffsets = 273;
constexpr quint16 kTiffTagOrientation = 274;
constexpr quint16 kTiffTagSamplesPerPixel = 277;
constexpr quint16 kTiffTagRowsPerStrip = 278;
constexpr quint16 kTiffTagStripByteCounts = 279;
constexpr quint16 kTiffTagPlanarConfiguration = 284;
constexpr quint16 kTiffTagSampleFormat = 339;
constexpr quint16 kTiffCompressionNone = 1;
constexpr quint16 kTiffPhotometricBlackIsZero = 1;
constexpr quint16 kTiffOrientationTopLeft = 1;
constexpr quint16 kTiffPlanarConfigurationContiguous = 1;
constexpr quint16 kTiffSampleFormatUnsignedInteger = 1;
constexpr quint32 kTiffIfdOffset = 8;
constexpr quint16 kTiffTagCount = 12;
constexpr quint32 kTiffPixelOffset = kTiffIfdOffset + sizeof(quint16)
    + static_cast<quint32>(kTiffTagCount) * 12U + sizeof(quint32);

bool isTiffPath(const QString& path)
{
    const QString suffix = QFileInfo(path).suffix();
    return suffix.compare(QStringLiteral("tif"), Qt::CaseInsensitive) == 0
        || suffix.compare(QStringLiteral("tiff"), Qt::CaseInsensitive) == 0;
}

bool ensureOutputDirectory(const QString& filePath, QString* errorMessage)
{
    const QString directory = QFileInfo(filePath).absolutePath();
    thread_local QString previousDirectory;
    if (directory == previousDirectory)
    {
        return true;
    }
    if (!QDir().mkpath(directory))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("failed to create output directory: %1").arg(directory);
        }
        return false;
    }
    previousDirectory = directory;
    return true;
}

void writeTiffTag(QDataStream& stream, quint16 tag, quint16 type, quint32 count, quint32 value)
{
    stream << tag << type << count << value;
}

bool writeUncompressedGray16Tiff(const QString& filePath, const cv::Mat& image, QString* errorMessage)
{
    if (image.type() != CV_16UC1 || image.cols <= 0 || image.rows <= 0)
    {
        return false;
    }

    const quint64 pixelBytes64 = static_cast<quint64>(image.cols)
        * static_cast<quint64>(image.rows) * sizeof(quint16);
    if (pixelBytes64 > std::numeric_limits<quint32>::max())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("uncompressed TIFF image exceeds the 4 GiB classic TIFF limit");
        }
        return false;
    }

    if (!ensureOutputDirectory(filePath, errorMessage))
    {
        return false;
    }
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("failed to open output file: %1").arg(file.errorString());
        }
        return false;
    }

    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream << static_cast<quint8>('I') << static_cast<quint8>('I')
           << static_cast<quint16>(42) << kTiffIfdOffset;
    stream << kTiffTagCount;
    const quint32 pixelBytes = static_cast<quint32>(pixelBytes64);
    writeTiffTag(stream, kTiffTagImageWidth, kTiffTypeLong, 1, static_cast<quint32>(image.cols));
    writeTiffTag(stream, kTiffTagImageLength, kTiffTypeLong, 1, static_cast<quint32>(image.rows));
    writeTiffTag(stream, kTiffTagBitsPerSample, kTiffTypeShort, 1, 16);
    writeTiffTag(stream, kTiffTagCompression, kTiffTypeShort, 1, kTiffCompressionNone);
    writeTiffTag(stream, kTiffTagPhotometricInterpretation, kTiffTypeShort, 1, kTiffPhotometricBlackIsZero);
    writeTiffTag(stream, kTiffTagStripOffsets, kTiffTypeLong, 1, kTiffPixelOffset);
    writeTiffTag(stream, kTiffTagOrientation, kTiffTypeShort, 1, kTiffOrientationTopLeft);
    writeTiffTag(stream, kTiffTagSamplesPerPixel, kTiffTypeShort, 1, 1);
    writeTiffTag(stream, kTiffTagRowsPerStrip, kTiffTypeLong, 1, static_cast<quint32>(image.rows));
    writeTiffTag(stream, kTiffTagStripByteCounts, kTiffTypeLong, 1, pixelBytes);
    writeTiffTag(stream, kTiffTagPlanarConfiguration, kTiffTypeShort, 1, kTiffPlanarConfigurationContiguous);
    writeTiffTag(stream, kTiffTagSampleFormat, kTiffTypeShort, 1, kTiffSampleFormatUnsignedInteger);
    stream << static_cast<quint32>(0);
    if (stream.status() != QDataStream::Ok)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("failed to write TIFF header");
        }
        return false;
    }

    const qint64 contiguousBytes = static_cast<qint64>(pixelBytes);
    if (image.isContinuous())
    {
        const qint64 written = file.write(reinterpret_cast<const char*>(image.ptr<quint16>(0)), contiguousBytes);
        if (written != contiguousBytes)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("short TIFF pixel write: %1 / %2 bytes")
                    .arg(written)
                    .arg(contiguousBytes);
            }
            return false;
        }
        return true;
    }

    const qint64 rowBytes = static_cast<qint64>(image.cols) * static_cast<qint64>(sizeof(quint16));
    for (int row = 0; row < image.rows; ++row)
    {
        const qint64 written = file.write(reinterpret_cast<const char*>(image.ptr<quint16>(row)), rowBytes);
        if (written != rowBytes)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("short TIFF pixel write: %1 / %2 bytes")
                    .arg(written)
                    .arg(rowBytes);
            }
            return false;
        }
    }
    return true;
}

} // namespace

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

    if (isTiffPath(filePath) && image.type() == CV_16UC1)
    {
        return writeUncompressedGray16Tiff(filePath, image, errorMessage);
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

    if (!ensureOutputDirectory(filePath, errorMessage))
    {
        return false;
    }
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
