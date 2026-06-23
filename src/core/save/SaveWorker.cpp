#include "core/save/SaveWorker.h"

#include "core/recon/IReconstructionVolume.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <future>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ct
{

namespace
{

bool saveImageWithQtPath(const QString& filePath, const cv::Mat& image)
{
    const QString extension = QStringLiteral(".") + QFileInfo(filePath).suffix();
    std::vector<uchar> encoded;
    std::vector<int> params;
    if (extension.compare(QStringLiteral(".tif"), Qt::CaseInsensitive) == 0
        || extension.compare(QStringLiteral(".tiff"), Qt::CaseInsensitive) == 0)
    {
        params = { cv::IMWRITE_TIFF_COMPRESSION, 1 };
    }
    if (!cv::imencode(extension.toStdString(), image, encoded, params))
    {
        return false;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }

    return file.write(reinterpret_cast<const char*>(encoded.data()),
                      static_cast<qint64>(encoded.size())) == static_cast<qint64>(encoded.size());
}

cv::Mat convertSliceForSave(const cv::Mat& slice, SaveDepth depth, float globalMin, float globalMax)
{
    if (depth == SaveDepth::Bit32)
    {
        return slice.clone();
    }

    const float range = (globalMax > globalMin) ? (globalMax - globalMin) : 1.0f;
    cv::Mat clipped;
    cv::min(cv::max(slice, globalMin), globalMax, clipped);
    cv::Mat output;
    if (depth == SaveDepth::Bit16)
    {
        clipped.convertTo(output, CV_16U, 65535.0 / range, -globalMin * 65535.0 / range);
    }
    else
    {
        clipped.convertTo(output, CV_8U, 255.0 / range, -globalMin * 255.0 / range);
    }
    return output;
}

} // namespace

SaveWorker::SaveWorker(SaveRequest request, QObject* parent)
    : QObject(parent)
    , request_(std::move(request))
{
}

void SaveWorker::requestStop()
{
    stopRequested_.store(true);
}

void SaveWorker::process()
{
    try
    {
        if (!request_.volume)
        {
            throw std::runtime_error("No volume data available.");
        }

        const QDir saveDir(request_.config.io.saveDir);
        if (!saveDir.exists())
        {
            QDir().mkpath(request_.config.io.saveDir);
        }

        request_.config.saveToJsonFile(saveDir.filePath("recon_params.json"));

        float globalMin = request_.useDisplayWindow ? request_.displayMin : 0.0f;
        float globalMax = request_.useDisplayWindow ? request_.displayMax : 0.0f;
        bool firstChunk = true;

        if (!request_.useDisplayWindow && request_.config.io.saveDepth != SaveDepth::Bit32)
        {
            for (int z = 0; z < request_.volume->depth() && !stopRequested_.load(); z += request_.config.compute.zChunkSize)
            {
                std::vector<float> hostChunk;
                const int count = std::min(request_.config.compute.zChunkSize, request_.volume->depth() - z);
                if (!request_.volume->downloadRange(z, count, hostChunk))
                {
                    throw std::runtime_error("Failed to download volume range for min/max.");
                }

                const auto [chunkMinIt, chunkMaxIt] = std::minmax_element(hostChunk.begin(), hostChunk.end());
                if (firstChunk)
                {
                    globalMin = *chunkMinIt;
                    globalMax = *chunkMaxIt;
                    firstChunk = false;
                }
                else
                {
                    globalMin = std::min(globalMin, *chunkMinIt);
                    globalMax = std::max(globalMax, *chunkMaxIt);
                }
            }
        }

        const int threadCount = std::clamp(request_.config.io.tiffThreads, 1, 64);
        emit logMessage(QStringLiteral("Saving TIFF slices with %1 writer threads, no compression.").arg(threadCount));
        std::vector<std::future<void>> futures;
        int completed = 0;
        auto waitOne = [&]() {
            futures.front().get();
            futures.erase(futures.begin());
            ++completed;
            emit progressUpdated(completed, request_.volume->depth());
        };

        for (int z = 0; z < request_.volume->depth() && !stopRequested_.load(); ++z)
        {
            const QString filePath = saveDir.filePath(
                QString("%1_%2.%3")
                    .arg(request_.config.io.filePrefix)
                    .arg(z, 4, 10, QLatin1Char('0'))
                    .arg(request_.config.io.saveFormat));
            auto volume = request_.volume;
            const SaveDepth saveDepth = request_.config.io.saveDepth;
            futures.push_back(std::async(std::launch::async, [volume, z, saveDepth, globalMin, globalMax, filePath]() {
                const cv::Mat slice = volume->downloadSlice(z);
                const cv::Mat output = convertSliceForSave(slice, saveDepth, globalMin, globalMax);
                if (!saveImageWithQtPath(filePath, output))
                {
                    throw std::runtime_error("Failed to save image slice.");
                }
            }));
            if (static_cast<int>(futures.size()) >= threadCount)
            {
                waitOne();
            }
        }
        while (!futures.empty())
        {
            waitOne();
        }

        if (stopRequested_.load())
        {
            emit logMessage(QStringLiteral(">>> Save stopped."));
        }
        emit saveFinished();
    }
    catch (const std::exception& ex)
    {
        emit saveFailed(QString::fromLocal8Bit(ex.what()));
    }
}

} // namespace ct
