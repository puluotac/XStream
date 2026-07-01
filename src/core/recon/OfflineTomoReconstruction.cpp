#include "core/recon/OfflineTomoReconstruction.h"

#include "core/cor/CorEstimatorFft.h"
#include "core/io/FileProjectionSource.h"
#include "core/preprocess/ImageNormalizer.h"
#include "core/preprocess/PaganinProcessor.h"
#include "core/recon/GpuProjectionBatch.h"
#include "cuda/offline_tomo_kernels.h"
#include "cuda/preprocess_kernels.h"
#include "cuda/stripe_removal_kernels.h"

#include <QDir>
#include <QElapsedTimer>
#include <QCoreApplication>
#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThread>
#include <QTextStream>

#include <cuda_runtime.h>
#include <cufft.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ct
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

void appendReconDebug(const QString& text)
{
    QFile file(QCoreApplication::applicationDirPath() + QStringLiteral("/ct_recon_debug.log"));
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        QTextStream stream(&file);
        stream << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << " " << text << '\n';
    }
}

QString formatSeconds(qint64 ms)
{
    return QString::number(ms / 1000.0, 'f', 3);
}

QString formatGiB(size_t bytes)
{
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    return QString::number(static_cast<double>(bytes) / gib, 'f', 2);
}

QString formatMiBPerSecond(qint64 bytes, qint64 ms)
{
    if (ms <= 0)
    {
        return QStringLiteral("n/a");
    }
    constexpr double mib = 1024.0 * 1024.0;
    return QString::number((static_cast<double>(bytes) / mib) / (static_cast<double>(ms) / 1000.0), 'f', 1);
}

void validateGeometryImageSize(const cv::Mat& image, const QString& label, const AppConfig& config)
{
    if (image.empty())
    {
        return;
    }
    if (image.rows != config.geometry.volH || image.cols != config.geometry.volW || image.channels() != 1)
    {
        throw std::runtime_error(QStringLiteral("%1 image size mismatch: got %2x%3x%4, expected %5x%6x1 from JSON geometry.")
                .arg(label)
                .arg(image.cols)
                .arg(image.rows)
                .arg(image.channels())
                .arg(config.geometry.volW)
                .arg(config.geometry.volH)
                .toStdString());
    }
}

void resolveOfflineGeometryFromConfigOrReferences(AppConfig& config,
    const cv::Mat& flat,
    const cv::Mat& dark,
    const cv::Mat& ref0,
    const cv::Mat& ref180,
    const OfflineTomoReconstruction::LogCallback& log)
{
    if (config.geometry.volW <= 0 || config.geometry.volH <= 0)
    {
        const cv::Mat* source = nullptr;
        QString sourceLabel;
        if (!flat.empty())
        {
            source = &flat;
            sourceLabel = QStringLiteral("flat");
        }
        else if (!ref0.empty())
        {
            source = &ref0;
            sourceLabel = QStringLiteral("0 deg reference");
        }
        else if (!ref180.empty())
        {
            source = &ref180;
            sourceLabel = QStringLiteral("180 deg reference");
        }
        else if (!dark.empty())
        {
            source = &dark;
            sourceLabel = QStringLiteral("dark");
        }

        if (source == nullptr)
        {
            throw std::runtime_error("Offline geometry VOL_W/VOL_H are missing and no reference image is available to infer them.");
        }
        config.geometry.volH = source->rows;
        config.geometry.volW = source->cols;
        log(QStringLiteral("Offline geometry inferred from %1: VOL_W=%2, VOL_H=%3.")
                .arg(sourceLabel)
                .arg(config.geometry.volW)
                .arg(config.geometry.volH));
    }
    else
    {
        log(QStringLiteral("Offline geometry uses JSON VOL_W=%1, VOL_H=%2; reference images are only used for correction/COR.")
                .arg(config.geometry.volW)
                .arg(config.geometry.volH));
    }

    validateGeometryImageSize(flat, QStringLiteral("Flat"), config);
    validateGeometryImageSize(dark, QStringLiteral("Dark"), config);
    validateGeometryImageSize(ref0, QStringLiteral("0 deg reference"), config);
    validateGeometryImageSize(ref180, QStringLiteral("180 deg reference"), config);
}

int paddedLengthForWidth(int width)
{
    int padded = 1;
    while (padded < width * 2)
    {
        padded <<= 1;
    }
    return padded;
}

std::vector<float> makeFbpFilter(int paddedLength, FilterType type)
{
    const int spectrumLength = paddedLength / 2 + 1;
    std::vector<float> filter(static_cast<size_t>(spectrumLength), 0.0f);
    for (int i = 0; i < spectrumLength; ++i)
    {
        const double freq = static_cast<double>(i) / paddedLength;
        const double w = freq * 2.0;
        double window = 1.0;
        switch (type)
        {
        case FilterType::RamLak:
            window = 1.0;
            break;
        case FilterType::SheppLogan:
            window = (freq == 0.0) ? 1.0 : std::sin(kPi * freq) / (kPi * freq);
            break;
        case FilterType::Cosine:
            window = std::cos(freq * kPi);
            break;
        case FilterType::Hamming:
            window = 0.54 + 0.46 * std::cos(w * kPi);
            break;
        case FilterType::Hann:
            window = 0.5 + 0.5 * std::cos(w * kPi);
            break;
        }
        filter[static_cast<size_t>(i)] = static_cast<float>(std::abs(freq) * window);
    }
    return filter;
}

std::vector<float> makeFbpGeometry(int angleCount, float angleOffsetDeg)
{
    std::vector<float> geom(static_cast<size_t>(angleCount) * 2, 0.0f);
    const double offset = angleOffsetDeg * kPi / 180.0;
    for (int i = 0; i < angleCount; ++i)
    {
        const double angle = i * (kPi / angleCount) + offset;
        geom[static_cast<size_t>(i) * 2] = static_cast<float>(std::sin(angle));
        geom[static_cast<size_t>(i) * 2 + 1] = static_cast<float>(-std::cos(angle));
    }
    return geom;
}

std::vector<double> makeGridrecVectors(int angleCount, float corShift, float angleOffsetDeg)
{
    std::vector<double> vectors(static_cast<size_t>(angleCount) * 12, 0.0);
    const double offset = angleOffsetDeg * kPi / 180.0;
    for (int i = 0; i < angleCount; ++i)
    {
        const double angle = i * (kPi / angleCount) + offset;
        const double rx = std::sin(angle);
        const double ry = -std::cos(angle);
        const double ux = std::cos(angle);
        const double uy = std::sin(angle);
        const size_t base = static_cast<size_t>(i) * 12;
        vectors[base + 0] = rx;
        vectors[base + 1] = ry;
        vectors[base + 3] = corShift * ux;
        vectors[base + 4] = corShift * uy;
        vectors[base + 6] = ux;
        vectors[base + 7] = uy;
        vectors[base + 11] = 1.0;
    }
    return vectors;
}

float i0Approx(float x)
{
    const float t = x * x / 14.0625f;
    return 1.0f + t * (3.5156229f + t * (3.0899424f + t * (1.2067492f
        + t * (0.2659732f + t * (0.0360768f + t * 0.0045813f)))));
}

std::vector<float> makeGridrecDeapodMap(int gridSize)
{
    std::vector<float> window(static_cast<size_t>(gridSize), 0.0f);
    const float beta = 6.0f;
    const float radius = 2.0f;
    for (int x = 0; x < gridSize; ++x)
    {
        const float centered = static_cast<float>(x - gridSize / 2);
        if (std::abs(centered) < radius)
        {
            const float argSq = std::max(0.0f, 1.0f - (centered / radius) * (centered / radius));
            window[static_cast<size_t>(x)] = i0Approx(beta * std::sqrt(argSq)) / i0Approx(beta);
        }
    }

    std::vector<int> shiftedIndices;
    std::vector<float> shiftedValues;
    shiftedIndices.reserve(8);
    shiftedValues.reserve(8);
    for (int n = 0; n < gridSize; ++n)
    {
        const int srcIndex = (n + gridSize / 2) % gridSize;
        const float value = window[static_cast<size_t>(srcIndex)];
        if (value != 0.0f)
        {
            shiftedIndices.push_back(n);
            shiftedValues.push_back(value);
        }
    }

    std::vector<float> ifftProfile(static_cast<size_t>(gridSize), 0.0f);
    for (int m = 0; m < gridSize; ++m)
    {
        double value = 0.0;
        for (size_t i = 0; i < shiftedIndices.size(); ++i)
        {
            const double phase = 2.0 * kPi * static_cast<double>(m) * shiftedIndices[i]
                / static_cast<double>(gridSize);
            value += static_cast<double>(shiftedValues[i]) * std::cos(phase);
        }
        ifftProfile[static_cast<size_t>(m)] = static_cast<float>(value / gridSize);
    }

    std::vector<float> profile(static_cast<size_t>(gridSize), 0.0f);
    float maxValue = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < gridSize; ++i)
    {
        const int srcIndex = (i + gridSize / 2) % gridSize;
        const float value = ifftProfile[static_cast<size_t>(srcIndex)];
        profile[static_cast<size_t>(i)] = value;
        maxValue = std::max(maxValue, value);
    }
    if (maxValue <= 0.0f || !std::isfinite(maxValue))
    {
        throw std::runtime_error("Failed to build Gridrec deapodization profile.");
    }
    for (float& value : profile)
    {
        value /= maxValue;
    }

    std::vector<float> map(static_cast<size_t>(gridSize) * gridSize, 1.0f);
    for (int y = 0; y < gridSize; ++y)
    {
        for (int x = 0; x < gridSize; ++x)
        {
            const float py = std::max(profile[static_cast<size_t>(y)], 1e-6f);
            const float px = std::max(profile[static_cast<size_t>(x)], 1e-6f);
            map[static_cast<size_t>(y) * gridSize + x] = std::max(py * px, 1e-6f);
        }
    }
    return map;
}

void ensureCuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

void ensureCufft(cufftResult status, const char* message)
{
    if (status != CUFFT_SUCCESS)
    {
        throw std::runtime_error(std::string(message) + " failed.");
    }
}

class DeviceFloatBuffer
{
public:
    DeviceFloatBuffer() = default;
    DeviceFloatBuffer(const DeviceFloatBuffer&) = delete;
    DeviceFloatBuffer& operator=(const DeviceFloatBuffer&) = delete;
    DeviceFloatBuffer(DeviceFloatBuffer&& other) noexcept
        : ptr_(other.ptr_)
        , count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceFloatBuffer& operator=(DeviceFloatBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceFloatBuffer()
    {
        release();
    }

    bool allocate(size_t count)
    {
        release();
        count_ = count;
        return cudaMalloc(&ptr_, count * sizeof(float)) == cudaSuccess;
    }

    void release()
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    float* ptr() { return ptr_; }
    const float* ptr() const { return ptr_; }
    size_t count() const { return count_; }

private:
    float* ptr_ = nullptr;
    size_t count_ = 0;
};

class DeviceUint16Buffer
{
public:
    ~DeviceUint16Buffer()
    {
        release();
    }

    bool allocate(size_t count)
    {
        release();
        count_ = count;
        return cudaMalloc(&ptr_, count * sizeof(uint16_t)) == cudaSuccess;
    }

    void release()
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    uint16_t* ptr() { return ptr_; }
    const uint16_t* ptr() const { return ptr_; }
    size_t count() const { return count_; }

private:
    uint16_t* ptr_ = nullptr;
    size_t count_ = 0;
};

class DeviceComplexBuffer
{
public:
    ~DeviceComplexBuffer()
    {
        release();
    }

    bool allocate(size_t count)
    {
        release();
        count_ = count;
        return cudaMalloc(&ptr_, count * sizeof(cufftComplex)) == cudaSuccess;
    }

    bool ensureCapacity(size_t count)
    {
        if (count <= count_ && ptr_ != nullptr)
        {
            return true;
        }
        return allocate(count);
    }

    void release()
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    cufftComplex* ptr() { return ptr_; }
    const cufftComplex* ptr() const { return ptr_; }

private:
    cufftComplex* ptr_ = nullptr;
    size_t count_ = 0;
};

class CufftHandle
{
public:
    ~CufftHandle()
    {
        destroy();
    }

    cufftHandle* out()
    {
        destroy();
        return &handle_;
    }

    cufftHandle get() const
    {
        return handle_;
    }

    void destroy()
    {
        if (handle_ != 0)
        {
            cufftDestroy(handle_);
            handle_ = 0;
        }
    }

private:
    cufftHandle handle_ = 0;
};

class CudaEvent
{
public:
    CudaEvent()
    {
        ensureCuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "create CUDA event");
    }

    explicit CudaEvent(bool enableTiming)
    {
        const unsigned int flags = enableTiming ? cudaEventDefault : cudaEventDisableTiming;
        ensureCuda(cudaEventCreateWithFlags(&event_, flags), "create CUDA event");
    }

    ~CudaEvent()
    {
        if (event_ != nullptr)
        {
            cudaEventDestroy(event_);
        }
    }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

struct ChunkSaveItem
{
    int zStart = 0;
    int zCount = 0;
    int width = 0;
    std::vector<uint16_t> data;
};

struct ChunkSaveStats
{
    qint64 savedSlices = 0;
    qint64 negativeClamped = 0;
    qint64 saturated = 0;
    qint64 finiteSamples = 0;
    qint64 bytesWritten = 0;
    qint64 saveMs = 0;
};

struct OfflineSaveWindow
{
    float vmin = 0.0f;
    float vmax = 1.0f;

    float range() const
    {
        return std::max(vmax - vmin, 1.0e-6f);
    }
};

constexpr double kAutoSaveWindowLowPercentile = 0.10;
constexpr double kAutoSaveWindowHighPercentile = 99.95;
constexpr double kAutoSaveWindowHighMarginFraction = 0.05;

struct OfflineInputBatch
{
    int batchIndex = 0;
    int validCount = 0;
    std::vector<int> logicalIndices;
};

struct ResolvedFlatGroup
{
    QString label;
    int afterProjectionCount = 0;
    int firstFlatIndex = 0;
    int flatCount = 0;
    cv::Mat flatAverage;
    DeviceCorrectionImages correctionImages;
};

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

class IChunkResultWriter
{
public:
    virtual ~IChunkResultWriter() = default;
    virtual void enqueue(ChunkSaveItem item) = 0;
    virtual void requestStop() = 0;
    virtual void finish() = 0;
    virtual ChunkSaveStats stats() const = 0;
    virtual QString outputDescription() const = 0;
};

struct TiffSliceTask
{
    std::shared_ptr<std::vector<uint16_t>> chunk;
    int localZ = 0;
    int globalZ = 0;
    int width = 0;
};

class TiffSlicePoolWriter : public IChunkResultWriter
{
public:
    TiffSlicePoolWriter(AppConfig config, OfflineSaveWindow saveWindow,
        const OfflineTomoReconstruction::LogCallback& log)
        : config_(std::move(config))
        , saveWindow_(saveWindow)
        , log_(log)
    {
        const int requestedThreads = config_.io.tiffThreads;
        config_.io.tiffThreads = std::clamp(requestedThreads, 1, 64);
        threadCount_ = config_.io.tiffThreads;
        if (saveWindow_.vmax <= saveWindow_.vmin)
        {
            saveWindow_.vmin = 0.0f;
            saveWindow_.vmax = 1.0f;
        }
        QDir saveDir(config_.io.saveDir);
        if (!saveDir.exists())
        {
            QDir().mkpath(config_.io.saveDir);
        }
        config_.io.saveDepth = SaveDepth::Bit16;
        if (config_.io.saveFormat.isEmpty())
        {
            config_.io.saveFormat = QStringLiteral("tiff");
        }
        config_.saveToJsonFile(saveDir.filePath(QStringLiteral("recon_params.json")));
        for (int i = 0; i < threadCount_; ++i)
        {
            workers_.emplace_back([this]() { runWorker(); });
        }
        log_(QStringLiteral("Offline save mode: TIFF Slices, %1 writer threads, no compression (requested %2).")
                .arg(threadCount_)
                .arg(requestedThreads));
    }

    ~TiffSlicePoolWriter() override
    {
        try
        {
            finish();
        }
        catch (...)
        {
        }
    }

    void enqueue(ChunkSaveItem item) override
    {
        auto chunk = std::make_shared<std::vector<uint16_t>>(std::move(item.data));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int localZ = 0; localZ < item.zCount; ++localZ)
            {
                TiffSliceTask task;
                task.chunk = chunk;
                task.localZ = localZ;
                task.globalZ = item.zStart + localZ;
                task.width = item.width;
                tasks_.push_back(std::move(task));
            }
        }
        cv_.notify_all();
    }

    void requestStop() override
    {
        stopRequested_.store(true);
        cv_.notify_all();
    }

    void finish() override
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (finished_)
            {
                return;
            }
            closed_ = true;
            finished_ = true;
        }
        cv_.notify_all();
        for (std::thread& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        if (!error_.empty())
        {
            throw std::runtime_error(error_);
        }
    }

    ChunkSaveStats stats() const override
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        return stats_;
    }

    QString outputDescription() const override
    {
        return config_.io.saveDir;
    }

private:
    void runWorker()
    {
        try
        {
            while (true)
            {
                TiffSliceTask task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this]() {
                        return closed_ || stopRequested_.load() || !tasks_.empty();
                    });
                    if ((closed_ || stopRequested_.load()) && tasks_.empty())
                    {
                        break;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                }
                saveTask(task);
            }
        }
        catch (const std::exception& ex)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (error_.empty())
                {
                    error_ = ex.what();
                }
            }
            stopRequested_.store(true);
            cv_.notify_all();
        }
    }

    void saveTask(const TiffSliceTask& task)
    {
        QElapsedTimer timer;
        timer.start();
        const size_t slicePixels = static_cast<size_t>(task.width) * task.width;
        const uint16_t* src = task.chunk->data() + static_cast<size_t>(task.localZ) * slicePixels;
        cv::Mat output(task.width, task.width, CV_16U, const_cast<uint16_t*>(src));
        qint64 belowMin = 0;
        qint64 saturated = 0;
        for (size_t i = 0; i < slicePixels; ++i)
        {
            belowMin += (src[i] == 0) ? 1 : 0;
            saturated += (src[i] == 65535) ? 1 : 0;
        }

        QDir saveDir(config_.io.saveDir);
        const QString filePath = saveDir.filePath(
            QString("%1_%2.%3")
                .arg(config_.io.filePrefix)
                .arg(task.globalZ, 4, 10, QLatin1Char('0'))
                .arg(config_.io.saveFormat));
        if (!saveImageWithQtPath(filePath, output))
        {
            throw std::runtime_error("Failed to save offline TIFF slice.");
        }

        std::lock_guard<std::mutex> lock(statsMutex_);
        ++stats_.savedSlices;
        stats_.negativeClamped += belowMin;
        stats_.saturated += saturated;
        stats_.finiteSamples += static_cast<qint64>(slicePixels);
        stats_.bytesWritten += QFileInfo(filePath).size();
        stats_.saveMs += timer.elapsed();
    }

    AppConfig config_;
    OfflineSaveWindow saveWindow_;
    OfflineTomoReconstruction::LogCallback log_;
    int threadCount_ = 1;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<TiffSliceTask> tasks_;
    std::vector<std::thread> workers_;
    std::atomic_bool stopRequested_ = false;
    bool closed_ = false;
    bool finished_ = false;
    std::string error_;
    mutable std::mutex statsMutex_;
    ChunkSaveStats stats_;
};

std::unique_ptr<IChunkResultWriter> makeChunkWriter(const AppConfig& config, const OfflineSaveWindow& saveWindow,
    const OfflineTomoReconstruction::LogCallback& log)
{
    return std::make_unique<TiffSlicePoolWriter>(config, saveWindow, log);
}

OfflineInputBatch loadOfflineInputBatch(FileProjectionSource& source, const AppConfig& config,
    HostPinnedProjectionBatch& hostBatch, int startIndex, int firstLogicalIndex,
    int batchIndex, std::atomic_bool& stopRequested)
{
    OfflineInputBatch batch;
    batch.batchIndex = batchIndex;
    batch.logicalIndices.reserve(static_cast<size_t>(config.compute.batchSize));
    hostBatch.clear();

    const int angleCount = config.geometry.totalAngles;
    const int height = config.geometry.volH;
    const int width = config.geometry.volW;
    const int endLogical = std::min(angleCount, firstLogicalIndex + config.compute.batchSize);
    QDir inputDir(config.io.watchDir);

    for (int logical = firstLogicalIndex; logical < endLogical && !stopRequested.load(); ++logical)
    {
        const int fileIndex = logical + startIndex;
        const QString filePath = inputDir.filePath(source.makeNumberedFileName(config.io.tomoPrefix, fileIndex));
        const cv::Mat image = source.readImageRobust(filePath);
        if (image.empty())
        {
            throw std::runtime_error(QStringLiteral("Offline projection file is missing or unreadable: %1")
                    .arg(filePath)
                    .toStdString());
        }
        if (image.rows != height || image.cols != width || image.channels() != 1)
        {
            throw std::runtime_error(QStringLiteral("Offline projection image size mismatch: %1")
                    .arg(filePath)
                    .toStdString());
        }
        if (!hostBatch.copyFrame(batch.validCount, image))
        {
            throw std::runtime_error("Failed to copy offline projection into pinned input batch.");
        }
        batch.logicalIndices.push_back(logical);
        ++batch.validCount;
    }

    return batch;
}

int findFirstExistingFlatIndex(FileProjectionSource& source, const AppConfig& config, std::atomic_bool& stopRequested)
{
    QDir inputDir(config.io.watchDir);
    for (int index = 0; index < 100000 && !stopRequested.load(); ++index)
    {
        const QString filePath = inputDir.filePath(source.makeNumberedFileName(config.io.flatPrefix, index));
        if (QFileInfo::exists(filePath))
        {
            return index;
        }
    }
    return -1;
}

std::vector<FlatGroupConfig> inferFlatGroupsFromConfig(FileProjectionSource& source, const AppConfig& config,
    std::atomic_bool& stopRequested)
{
    std::vector<FlatGroupConfig> groups;
    if (config.io.correctionMode == CorrectionMode::None)
    {
        return groups;
    }
    const int flatNum = std::max(1, config.preprocess.flatNum);
    const int firstFlatIndex = findFirstExistingFlatIndex(source, config, stopRequested);
    if (firstFlatIndex < 0)
    {
        throw std::runtime_error("Flat correction requested but no flat file matching FLAT_PREFIX was found.");
    }

    auto appendGroup = [&](const QString& label, int afterProjectionCount) {
        FlatGroupConfig group;
        group.label = label;
        group.afterProjectionCount = std::clamp(afterProjectionCount, 0, config.geometry.totalAngles);
        group.firstFlatIndex = firstFlatIndex + static_cast<int>(groups.size()) * flatNum;
        group.flatCount = flatNum;
        groups.push_back(group);
    };

    appendGroup(QStringLiteral("start"), 0);
    if (config.preprocess.flatGap > 0)
    {
        for (int after = config.preprocess.flatGap; after < config.geometry.totalAngles; after += config.preprocess.flatGap)
        {
            appendGroup(QStringLiteral("interval"), after);
        }
        appendGroup(QStringLiteral("end"), config.geometry.totalAngles);
    }
    return groups;
}

std::vector<FlatGroupConfig> flatGroupConfigsFromManifest(const AppConfig& config,
    const OfflineTomoReconstruction::LogCallback& log)
{
    std::vector<FlatGroupConfig> groups;
    if (config.io.watchDir.isEmpty())
    {
        return groups;
    }

    const QString manifestPath = QDir(config.io.watchDir).filePath(QStringLiteral("scan_manifest.json"));
    QFile file(manifestPath);
    if (!file.exists())
    {
        return groups;
    }
    if (!file.open(QIODevice::ReadOnly))
    {
        log(QStringLiteral("Offline flat groups: ignoring unreadable scan_manifest.json: %1.").arg(manifestPath));
        return groups;
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    {
        log(QStringLiteral("Offline flat groups: ignoring invalid scan_manifest.json: %1.").arg(parseError.errorString()));
        return groups;
    }

    const QJsonObject root = doc.object();
    if (!root.value(QStringLiteral("COMPLETED")).toBool(false))
    {
        log(QStringLiteral("Offline flat groups: scan_manifest.json exists but COMPLETED=false; falling back to UI/job flat settings."));
        return groups;
    }

    const QJsonArray flatGroups = root.value(QStringLiteral("FLAT_GROUPS")).toArray();
    groups.reserve(static_cast<size_t>(flatGroups.size()));
    for (const QJsonValue& value : flatGroups)
    {
        const QJsonObject obj = value.toObject();
        FlatGroupConfig group;
        group.label = obj.value(QStringLiteral("LABEL")).toString();
        group.afterProjectionCount = obj.value(QStringLiteral("AFTER_PROJECTION_COUNT")).toInt();
        group.firstFlatIndex = obj.value(QStringLiteral("FIRST_FLAT_INDEX")).toInt();
        group.flatCount = obj.value(QStringLiteral("FLAT_COUNT")).toInt();
        if (group.flatCount > 0)
        {
            groups.push_back(group);
        }
    }
    if (!groups.empty())
    {
        log(QStringLiteral("Offline flat groups: using %1 groups from scan_manifest.json.").arg(groups.size()));
    }
    return groups;
}

std::vector<FlatGroupConfig> flatGroupConfigsForRun(FileProjectionSource& source, const AppConfig& config,
    std::atomic_bool& stopRequested, const OfflineTomoReconstruction::LogCallback& log)
{
    if (config.io.correctionMode == CorrectionMode::None)
    {
        return {};
    }
    std::vector<FlatGroupConfig> groups;
    groups = flatGroupConfigsFromManifest(config, log);
    if (groups.empty())
    {
        if (!config.preprocess.flatGroups.isEmpty())
        {
            groups.reserve(static_cast<size_t>(config.preprocess.flatGroups.size()));
            for (const FlatGroupConfig& group : config.preprocess.flatGroups)
            {
                if (group.flatCount > 0)
                {
                    groups.push_back(group);
                }
            }
            log(QStringLiteral("Offline flat groups: using %1 groups from recon job.").arg(groups.size()));
        }
        else
        {
            groups = inferFlatGroupsFromConfig(source, config, stopRequested);
            log(QStringLiteral("Offline flat groups: inferred %1 groups from Flat Gap=%2, Flat Num=%3.")
                    .arg(groups.size())
                    .arg(config.preprocess.flatGap)
                    .arg(config.preprocess.flatNum));
        }
    }
    std::sort(groups.begin(), groups.end(), [](const FlatGroupConfig& a, const FlatGroupConfig& b) {
        if (a.afterProjectionCount != b.afterProjectionCount)
        {
            return a.afterProjectionCount < b.afterProjectionCount;
        }
        return a.firstFlatIndex < b.firstFlatIndex;
    });
    return groups;
}

cv::Mat averageFlatGroup(FileProjectionSource& source, const AppConfig& config, const FlatGroupConfig& group,
    std::atomic_bool& stopRequested)
{
    const int height = config.geometry.volH;
    const int width = config.geometry.volW;
    cv::Mat sum(height, width, CV_32F, cv::Scalar(0.0f));
    QDir inputDir(config.io.watchDir);
    int loaded = 0;
    for (int i = 0; i < group.flatCount && !stopRequested.load(); ++i)
    {
        const int fileIndex = group.firstFlatIndex + i;
        const QString filePath = inputDir.filePath(source.makeNumberedFileName(config.io.flatPrefix, fileIndex));
        cv::Mat image = source.readImageRobust(filePath);
        if (image.empty())
        {
            throw std::runtime_error(QStringLiteral("Flat group '%1' is missing flat file: %2")
                    .arg(group.label.isEmpty() ? QString::number(group.afterProjectionCount) : group.label)
                    .arg(filePath)
                    .toStdString());
        }
        if (image.rows != height || image.cols != width || image.channels() != 1)
        {
            throw std::runtime_error(QStringLiteral("Flat group image size mismatch: %1")
                    .arg(filePath)
                    .toStdString());
        }
        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32F);
        sum += floatImage;
        ++loaded;
    }
    if (loaded <= 0)
    {
        throw std::runtime_error("Flat group has no readable flat images.");
    }
    sum *= (1.0f / static_cast<float>(loaded));
    return sum;
}

std::vector<ResolvedFlatGroup> loadResolvedFlatGroups(FileProjectionSource& source, AppConfig& config,
    const cv::Mat& dark, std::atomic_bool& stopRequested, const OfflineTomoReconstruction::LogCallback& log)
{
    std::vector<ResolvedFlatGroup> resolved;
    const std::vector<FlatGroupConfig> groupConfigs = flatGroupConfigsForRun(source, config, stopRequested, log);
    resolved.reserve(groupConfigs.size());
    for (const FlatGroupConfig& groupConfig : groupConfigs)
    {
        resolved.emplace_back();
        ResolvedFlatGroup& group = resolved.back();
        group.label = groupConfig.label;
        group.afterProjectionCount = groupConfig.afterProjectionCount;
        group.firstFlatIndex = groupConfig.firstFlatIndex;
        group.flatCount = groupConfig.flatCount;
        group.flatAverage = averageFlatGroup(source, config, groupConfig, stopRequested);
        if (!ImageNormalizer::uploadCorrectionImages(group.flatAverage, dark,
                config.geometry.volH, config.geometry.volW, group.correctionImages))
        {
            throw std::runtime_error("Failed to upload averaged flat group to GPU.");
        }
        log(QStringLiteral("Offline flat group %1: after=%2, firstFlat=%3, count=%4.")
                .arg(resolved.size())
                .arg(group.afterProjectionCount)
                .arg(group.firstFlatIndex)
                .arg(group.flatCount));
    }
    return resolved;
}

void releaseResolvedFlatGroups(std::vector<ResolvedFlatGroup>& groups)
{
    for (ResolvedFlatGroup& group : groups)
    {
        group.correctionImages.release();
    }
}

int nearestFlatGroupIndex(const std::vector<ResolvedFlatGroup>& groups, int projectionOrdinal)
{
    if (groups.empty())
    {
        return -1;
    }
    int best = 0;
    int bestDistance = std::abs(projectionOrdinal - groups.front().afterProjectionCount);
    for (int i = 1; i < static_cast<int>(groups.size()); ++i)
    {
        const int distance = std::abs(projectionOrdinal - groups[static_cast<size_t>(i)].afterProjectionCount);
        if (distance <= bestDistance)
        {
            best = i;
            bestDistance = distance;
        }
    }
    return best;
}

void preprocessAllProjections(AppConfig& config, FileProjectionSource& source, const cv::Mat& flat, const cv::Mat& dark,
    std::vector<float>& projectionStack, std::atomic_bool& stopRequested,
    const OfflineTomoReconstruction::LogCallback& log,
    const OfflineTomoReconstruction::ProgressCallback& progress)
{
    const int angleCount = config.geometry.totalAngles;
    const int height = config.geometry.volH;
    const int width = config.geometry.volW;
    (void)flat;
    const size_t projectionElements = static_cast<size_t>(angleCount) * height * width;
    projectionStack.assign(projectionElements, 0.0f);

    HostPinnedProjectionBatch hostBatches[2];
    GpuProjectionBatch gpuBatch;
    PaganinProcessor paganin;
    cudaStream_t stream = nullptr;

    if (!hostBatches[0].initialize(config.compute.batchSize, height, width)
        || !hostBatches[1].initialize(config.compute.batchSize, height, width)
        || !gpuBatch.initialize(config.compute.batchSize, height, width))
    {
        throw std::runtime_error("Failed to allocate offline preprocessing batch buffers.");
    }
    std::vector<ResolvedFlatGroup> flatGroups = loadResolvedFlatGroups(source, config, dark, stopRequested, log);
    if (flatGroups.empty() && (config.io.correctionMode == CorrectionMode::Flat || config.io.correctionMode == CorrectionMode::FlatDark))
    {
        throw std::runtime_error("Flat correction requested but no flat groups were resolved.");
    }
    paganin.configure(config.phase);
    if (paganin.isEnabled())
    {
        log(QStringLiteral("Offline Paganin phase retrieval enabled: GPU batch after flat/dark correction, before log/taper. Energy=%1 keV, distance=%2 m, pixel=%3 um, alpha=%4.")
            .arg(config.phase.energyKev, 0, 'g', 6)
            .arg(config.phase.distanceM, 0, 'g', 6)
            .arg(config.phase.pixelSizeUm, 0, 'g', 6)
            .arg(config.phase.alphaRatio, 0, 'g', 6));
    }
    ensureCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create offline preprocess stream");

    const int startIndex = source.extractStartIndex(config.io.filePattern0);
    int processed = 0;
    int batchIndex = 0;
    int nextLogical = 0;
    int pendingSlot = 0;
    log(QStringLiteral("Offline preprocess uses double-buffered pinned input: CPU TIFF read/decode overlaps GPU preprocessing."));
    auto launchLoad = [&](int slot, int firstLogical, int index) {
        return std::async(std::launch::async, [&source, &config, &hostBatches, &stopRequested, startIndex, slot, firstLogical, index]() {
            return loadOfflineInputBatch(source, config, hostBatches[slot], startIndex, firstLogical, index, stopRequested);
        });
    };

    std::future<OfflineInputBatch> pendingLoad = launchLoad(pendingSlot, nextLogical, batchIndex);
    nextLogical += config.compute.batchSize;
    ++batchIndex;

    while (processed < angleCount && !stopRequested.load())
    {
        OfflineInputBatch inputBatch = pendingLoad.get();
        const int currentSlot = pendingSlot;
        if (inputBatch.validCount <= 0)
        {
            continue;
        }

        if (nextLogical < angleCount)
        {
            pendingSlot = 1 - currentSlot;
            pendingLoad = launchLoad(pendingSlot, nextLogical, batchIndex);
            nextLogical += config.compute.batchSize;
            ++batchIndex;
        }

        const int validCount = inputBatch.validCount;
        log(QStringLiteral("Offline preprocess batch %1: %2 projections.").arg(inputBatch.batchIndex + 1).arg(validCount));
        if (!gpuBatch.uploadRawFromHostAsync(hostBatches[currentSlot], validCount, stream))
        {
            throw std::runtime_error("Failed to upload offline projection batch.");
        }
        if (config.io.correctionMode != CorrectionMode::None)
        {
            int segmentStart = 0;
            int currentGroupIndex = nearestFlatGroupIndex(flatGroups, inputBatch.logicalIndices.front() + 1);
            for (int localAngle = 1; localAngle <= validCount; ++localAngle)
            {
                const int nextGroupIndex = (localAngle < validCount)
                    ? nearestFlatGroupIndex(flatGroups, inputBatch.logicalIndices[static_cast<size_t>(localAngle)] + 1)
                    : -1;
                if (localAngle == validCount || nextGroupIndex != currentGroupIndex)
                {
                    if (currentGroupIndex < 0)
                    {
                        throw std::runtime_error("Offline flat/dark correction has no flat group for projection batch.");
                    }
                    const int segmentCount = localAngle - segmentStart;
                    if (!ImageNormalizer::applyCorrectionRange(gpuBatch, segmentStart, segmentCount,
                            flatGroups[static_cast<size_t>(currentGroupIndex)].correctionImages,
                            config.io.correctionMode, stream))
                    {
                        throw std::runtime_error("Offline flat/dark correction failed.");
                    }
                    segmentStart = localAngle;
                    currentGroupIndex = nextGroupIndex;
                }
            }
        }
        if (paganin.isEnabled() && !paganin.apply(gpuBatch, stream))
        {
            throw std::runtime_error("Offline Paganin phase retrieval failed.");
        }
        if (!ImageNormalizer::transposeRawToRecon(gpuBatch, stream))
        {
            throw std::runtime_error("Offline transpose failed.");
        }
        if (!ImageNormalizer::applyMethodAndTaper(gpuBatch, config.preprocess, stream))
        {
            throw std::runtime_error("Offline method/taper failed.");
        }
        ensureCuda(cudaStreamSynchronize(stream), "offline preprocess sync");

        std::vector<float> reconLayout(static_cast<size_t>(height) * validCount * width);
        ensureCuda(cudaMemcpy(reconLayout.data(), gpuBatch.reconLayout(),
            reconLayout.size() * sizeof(float), cudaMemcpyDeviceToHost), "download offline preprocessed batch");

        for (int localAngle = 0; localAngle < validCount; ++localAngle)
        {
            const int logical = inputBatch.logicalIndices[static_cast<size_t>(localAngle)];
            for (int z = 0; z < height; ++z)
            {
                const float* src = reconLayout.data() + (static_cast<size_t>(z) * validCount + localAngle) * width;
                float* dst = projectionStack.data() + (static_cast<size_t>(logical) * height + z) * width;
                std::memcpy(dst, src, static_cast<size_t>(width) * sizeof(float));
            }
        }
        processed += validCount;
        progress(processed, angleCount);
    }

    cudaStreamDestroy(stream);
    releaseResolvedFlatGroups(flatGroups);

    if (stopRequested.load())
    {
        return;
    }
    if (processed < angleCount)
    {
        throw std::runtime_error("Offline preprocessing did not receive all projection angles.");
    }
}

void fillSinogramChunk(const std::vector<float>& projectionStack, int angleCount, int height, int width,
    int zStart, int zCount, std::vector<float>& out)
{
    out.resize(static_cast<size_t>(zCount) * angleCount * width);
    for (int z = 0; z < zCount; ++z)
    {
        for (int a = 0; a < angleCount; ++a)
        {
            const float* src = projectionStack.data() + (static_cast<size_t>(a) * height + (zStart + z)) * width;
            float* dst = out.data() + (static_cast<size_t>(z) * angleCount + a) * width;
            std::memcpy(dst, src, static_cast<size_t>(width) * sizeof(float));
        }
    }
}

float percentileFinite(std::vector<float>& values, double percentile)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](float value) {
        return !std::isfinite(value);
    }), values.end());
    if (values.empty())
    {
        return 0.0f;
    }
    const double clamped = std::clamp(percentile, 0.0, 100.0);
    const size_t index = static_cast<size_t>(std::floor((clamped / 100.0) * (values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

OfflineSaveWindow makeOfflineSaveWindowFromCalibration(const std::vector<float>& calibration)
{
    std::vector<float> lowCalibration = calibration;
    std::vector<float> highCalibration = calibration;

    OfflineSaveWindow saveWindow;
    saveWindow.vmin = percentileFinite(lowCalibration, kAutoSaveWindowLowPercentile);
    saveWindow.vmax = percentileFinite(highCalibration, kAutoSaveWindowHighPercentile);
    if (saveWindow.vmax <= saveWindow.vmin)
    {
        saveWindow.vmax = saveWindow.vmin + 1.0f;
    }

    const float margin = saveWindow.range() * static_cast<float>(kAutoSaveWindowHighMarginFraction);
    saveWindow.vmax += margin;
    return saveWindow;
}

std::vector<int> makeCenterFirstChunkStarts(int height, int zChunk)
{
    std::vector<int> starts;
    for (int z = 0; z < height; z += zChunk)
    {
        starts.push_back(z);
    }
    if (starts.empty())
    {
        return starts;
    }
    const int centerZ = height / 2;
    auto centerIt = std::min_element(starts.begin(), starts.end(), [=](int lhs, int rhs) {
        return std::abs((lhs + std::min(zChunk, height - lhs) / 2) - centerZ)
            < std::abs((rhs + std::min(zChunk, height - rhs) / 2) - centerZ);
    });
    std::rotate(starts.begin(), centerIt, centerIt + 1);
    return starts;
}

cv::Mat makePreviewSliceFromChunk(const std::vector<float>& chunk, int zCount, int width)
{
    if (chunk.empty() || zCount <= 0 || width <= 0)
    {
        return {};
    }
    const int localZ = zCount / 2;
    const size_t slicePixels = static_cast<size_t>(width) * width;
    const float* src = chunk.data() + static_cast<size_t>(localZ) * slicePixels;
    cv::Mat slice(width, width, CV_32F);
    std::memcpy(slice.data, src, slicePixels * sizeof(float));
    return slice;
}

cv::Mat makePreviewSliceFromChunk(const std::vector<uint16_t>& chunk, int zCount, int width,
    const OfflineSaveWindow& saveWindow)
{
    if (chunk.empty() || zCount <= 0 || width <= 0)
    {
        return {};
    }
    const int localZ = zCount / 2;
    const size_t slicePixels = static_cast<size_t>(width) * width;
    const uint16_t* src = chunk.data() + static_cast<size_t>(localZ) * slicePixels;
    cv::Mat slice(width, width, CV_32F);
    float* dst = reinterpret_cast<float*>(slice.data);
    const float scale = saveWindow.range() / 65535.0f;
    for (size_t i = 0; i < slicePixels; ++i)
    {
        dst[i] = saveWindow.vmin + static_cast<float>(src[i]) * scale;
    }
    return slice;
}

void applyStripeRemovalFwIfEnabled(const AppConfig& config, float* deviceSino,
    int zCount, int angleCount, int width, cudaStream_t stream,
    const OfflineTomoReconstruction::LogCallback& log)
{
    if (!config.stripe.enable)
    {
        return;
    }

    if (config.stripe.method.compare(QStringLiteral("FW"), Qt::CaseInsensitive) != 0)
    {
        log(QStringLiteral("Stripe removal method '%1' is not implemented in C++; using FW db5.")
            .arg(config.stripe.method));
    }

    const int sliceChunk = std::clamp(config.stripe.chunk, 1, std::max(1, zCount));
    const float sigma = std::max(0.01f, config.stripe.sigma);
    const int autoLevel = static_cast<int>(std::ceil(std::log2(static_cast<double>(std::max({ zCount, angleCount, width })))));
    const int requestedLevel = config.stripe.level > 0 ? config.stripe.level : std::min(autoLevel, 4);
    const int level = std::clamp(requestedLevel, 1, 20);
    const int paddedAngleCount = config.stripe.pad ? (angleCount + angleCount / 8) : angleCount;
    const int angleShift = (paddedAngleCount - angleCount) / 2;

    DeviceFloatBuffer current;
    DeviceFloatBuffer nextApprox;
    DeviceFloatBuffer tempA;
    DeviceFloatBuffer tempD;
    DeviceFloatBuffer horiz;
    DeviceFloatBuffer low;
    DeviceFloatBuffer work;
    DeviceComplexBuffer spectrum;
    const size_t fullCount = static_cast<size_t>(sliceChunk) * paddedAngleCount * width;
    if (!current.allocate(fullCount) || !nextApprox.allocate(fullCount)
        || !tempA.allocate(fullCount) || !tempD.allocate(fullCount)
        || !horiz.allocate(fullCount) || !low.allocate(fullCount)
        || !work.allocate(fullCount))
    {
        throw std::runtime_error("Failed to allocate FW stripe removal GPU buffers.");
    }

    struct FwLevel
    {
        int angleLen = 0;
        int widthLen = 0;
        int angleCoeffLen = 0;
        int widthCoeffLen = 0;
        DeviceFloatBuffer cH;
        DeviceFloatBuffer cV;
        DeviceFloatBuffer cD;
    };

    const QString stripeMessage = QStringLiteral("FW stripe removal running: z=%1, angles=%2, width=%3, level=%4, sliceChunk=%5, sigma=%6.")
        .arg(zCount)
        .arg(angleCount)
        .arg(width)
        .arg(level)
        .arg(sliceChunk)
        .arg(sigma, 0, 'g', 6);
    log(stripeMessage);
    appendReconDebug(stripeMessage);

    for (int zOffset = 0; zOffset < zCount; zOffset += sliceChunk)
    {
        const int currZ = std::min(sliceChunk, zCount - zOffset);
        appendReconDebug(QStringLiteral("FW stripe zOffset=%1 currZ=%2").arg(zOffset).arg(currZ));
        float* currSino = deviceSino + static_cast<size_t>(zOffset) * angleCount * width;
        if (!launchPadStripeInput(currSino, current.ptr(), currZ, angleCount, paddedAngleCount,
                width, angleShift, stream))
        {
            throw std::runtime_error("FW stripe pad input failed.");
        }

        std::vector<FwLevel> levels;
        levels.reserve(static_cast<size_t>(level));
        int currentAngleLen = paddedAngleCount;
        int currentWidthLen = width;
        float* currentPtr = current.ptr();
        for (int lev = 0; lev < level; ++lev)
        {
            const int angleCoeffLen = (currentAngleLen + 9) / 2;
            const int anglePaddedLen = angleCoeffLen * 2;
            const int widthCoeffLen = (currentWidthLen + 9) / 2;
            const int widthPaddedLen = widthCoeffLen * 2;
            FwLevel fw;
            fw.angleLen = currentAngleLen;
            fw.widthLen = currentWidthLen;
            fw.angleCoeffLen = angleCoeffLen;
            fw.widthCoeffLen = widthCoeffLen;
            const size_t bandCount = static_cast<size_t>(currZ) * angleCoeffLen * widthCoeffLen;
            if (!fw.cH.allocate(bandCount) || !fw.cV.allocate(bandCount) || !fw.cD.allocate(bandCount))
            {
                throw std::runtime_error("Failed to allocate FW stripe wavelet bands.");
            }
            if (!launchDwtAxis1Db5(currentPtr, tempA.ptr(), tempD.ptr(),
                    currZ, currentAngleLen, currentWidthLen, angleCoeffLen, anglePaddedLen, stream))
            {
                throw std::runtime_error("FW stripe DWT axis-angle failed.");
            }
            if (!launchDwtAxis2Db5(tempA.ptr(), nextApprox.ptr(), fw.cV.ptr(),
                    currZ, angleCoeffLen, currentWidthLen, widthCoeffLen, widthPaddedLen, stream))
            {
                throw std::runtime_error("FW stripe DWT low detector failed.");
            }
            if (!launchDwtAxis2Db5(tempD.ptr(), fw.cH.ptr(), fw.cD.ptr(),
                    currZ, angleCoeffLen, currentWidthLen, widthCoeffLen, widthPaddedLen, stream))
            {
                throw std::runtime_error("FW stripe DWT high detector failed.");
            }
            levels.push_back(std::move(fw));
            std::swap(current, nextApprox);
            currentPtr = current.ptr();
            currentAngleLen = angleCoeffLen;
            currentWidthLen = widthCoeffLen;
            if (currentAngleLen <= 1 || currentWidthLen <= 1)
            {
                break;
            }
        }

        for (FwLevel& fw : levels)
        {
            const int spectrumLength = fw.angleCoeffLen / 2 + 1;
            const int vectorCount = currZ * fw.widthCoeffLen;
            const size_t spectrumCount = static_cast<size_t>(vectorCount) * spectrumLength;
            if (!spectrum.ensureCapacity(spectrumCount))
            {
                throw std::runtime_error("Failed to allocate FW stripe spectrum buffer.");
            }
            if (!launchPackSinogramColumnsForStripe(fw.cV.ptr(), work.ptr(),
                    currZ, fw.angleCoeffLen, fw.widthCoeffLen, stream))
            {
                throw std::runtime_error("FW stripe pack vertical detail failed.");
            }

            CufftHandle r2cPlan;
            CufftHandle c2rPlan;
            int n[1] = { fw.angleCoeffLen };
            int inembed[1] = { fw.angleCoeffLen };
            int onembed[1] = { spectrumLength };
            ensureCufft(cufftPlanMany(r2cPlan.out(), 1, n,
                inembed, 1, fw.angleCoeffLen,
                onembed, 1, spectrumLength,
                CUFFT_R2C, vectorCount), "FW stripe vertical detail R2C plan");
            ensureCufft(cufftPlanMany(c2rPlan.out(), 1, n,
                onembed, 1, spectrumLength,
                inembed, 1, fw.angleCoeffLen,
                CUFFT_C2R, vectorCount), "FW stripe vertical detail C2R plan");
            cufftSetStream(r2cPlan.get(), stream);
            cufftSetStream(c2rPlan.get(), stream);
            ensureCufft(cufftExecR2C(r2cPlan.get(),
                reinterpret_cast<cufftReal*>(work.ptr()), spectrum.ptr()), "FW stripe vertical detail R2C");
            if (!launchDampStripeSpectrum(spectrum.ptr(), vectorCount, spectrumLength,
                    fw.angleCoeffLen, sigma, stream))
            {
                throw std::runtime_error("FW stripe vertical detail spectrum damping failed.");
            }
            ensureCufft(cufftExecC2R(c2rPlan.get(), spectrum.ptr(),
                reinterpret_cast<cufftReal*>(work.ptr())), "FW stripe vertical detail C2R");
            if (!launchUnpackSinogramColumnsForStripe(work.ptr(), fw.cV.ptr(),
                    currZ, fw.angleCoeffLen, fw.widthCoeffLen, stream))
            {
                throw std::runtime_error("FW stripe unpack vertical detail failed.");
            }
        }

        currentPtr = current.ptr();
        for (int lev = static_cast<int>(levels.size()) - 1; lev >= 0; --lev)
        {
            FwLevel& fw = levels[static_cast<size_t>(lev)];
            if (!launchIdwtAxis2Db5(fw.cH.ptr(), fw.cD.ptr(), horiz.ptr(),
                    currZ, fw.angleCoeffLen, fw.widthCoeffLen, fw.widthLen, stream))
            {
                throw std::runtime_error("FW stripe IDWT detector high failed.");
            }
            if (!launchIdwtAxis2Db5(currentPtr, fw.cV.ptr(), low.ptr(),
                    currZ, fw.angleCoeffLen, fw.widthCoeffLen, fw.widthLen, stream))
            {
                throw std::runtime_error("FW stripe IDWT detector low failed.");
            }
            if (!launchIdwtAxis1Db5(low.ptr(), horiz.ptr(), work.ptr(),
                    currZ, fw.angleCoeffLen, fw.widthLen, fw.angleLen, stream))
            {
                throw std::runtime_error("FW stripe IDWT angle failed.");
            }
            std::swap(current, work);
            currentPtr = current.ptr();
        }

        if (!launchCropStripeOutput(currentPtr, currSino, currZ, angleCount,
                paddedAngleCount, width, angleShift, stream))
        {
            throw std::runtime_error("FW stripe crop output failed.");
        }
    }

    ensureCuda(cudaStreamSynchronize(stream), "FW stripe removal sync");
}

qint64 runOfflineFbp(const AppConfig& config, const std::vector<float>& projectionStack,
    std::atomic_bool& stopRequested,
    const OfflineTomoReconstruction::LogCallback& log,
    const OfflineTomoReconstruction::ProgressCallback& progress,
    const OfflineTomoReconstruction::DirectPreviewCallback& directPreview)
{
    QElapsedTimer reconstructionWallTimer;
    reconstructionWallTimer.start();
    const int angleCount = config.geometry.totalAngles;
    const int height = config.geometry.volH;
    const int width = config.geometry.volW;
    const int zChunk = std::min(config.compute.zChunkSize, height);
    const int paddedLength = paddedLengthForWidth(width);
    const int freqWidth = paddedLength / 2 + 1;

    DeviceFloatBuffer deviceSino;
    DeviceFloatBuffer devicePadded;
    DeviceComplexBuffer deviceSpectrum;
    DeviceFloatBuffer deviceFilter;
    DeviceFloatBuffer deviceOut;
    DeviceUint16Buffer deviceOut16;
    float2* deviceGeom = nullptr;
    cudaStream_t stream = nullptr;
    CufftHandle r2cPlan;
    CufftHandle c2rPlan;

    const size_t maxSino = static_cast<size_t>(zChunk) * angleCount * width;
    const size_t maxPadded = static_cast<size_t>(zChunk) * angleCount * paddedLength;
    const size_t maxSpectrum = static_cast<size_t>(zChunk) * angleCount * freqWidth;
    const size_t maxOut = static_cast<size_t>(zChunk) * width * width;

    if (!deviceSino.allocate(maxSino) || !devicePadded.allocate(maxPadded)
        || !deviceSpectrum.allocate(maxSpectrum) || !deviceFilter.allocate(freqWidth)
        || !deviceOut.allocate(maxOut) || !deviceOut16.allocate(maxOut))
    {
        throw std::runtime_error("Failed to allocate offline FBP GPU buffers.");
    }

    const auto filter = makeFbpFilter(paddedLength, config.preprocess.filterType);
    ensureCuda(cudaMemcpy(deviceFilter.ptr(), filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice),
        "upload offline FBP filter");

    const auto geom = makeFbpGeometry(angleCount, config.geometry.angleOffsetDeg);
    ensureCuda(cudaMalloc(&deviceGeom, static_cast<size_t>(angleCount) * sizeof(float2)), "allocate offline FBP geometry");
    ensureCuda(cudaMemcpy(deviceGeom, geom.data(), static_cast<size_t>(angleCount) * sizeof(float2), cudaMemcpyHostToDevice),
        "upload offline FBP geometry");
    ensureCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create offline FBP stream");

    const int rank = 1;
    int n[1] = { paddedLength };
    int inembed[1] = { paddedLength };
    int onembed[1] = { freqWidth };
    const int maxTransformCount = zChunk * angleCount;
    ensureCufft(cufftPlanMany(r2cPlan.out(), rank, n,
        inembed, 1, paddedLength,
        onembed, 1, freqWidth,
        CUFFT_R2C, maxTransformCount), "offline FBP R2C plan");
    ensureCufft(cufftPlanMany(c2rPlan.out(), rank, n,
        onembed, 1, freqWidth,
        inembed, 1, paddedLength,
        CUFFT_C2R, maxTransformCount), "offline FBP C2R plan");
    cufftSetStream(r2cPlan.get(), stream);
    cufftSetStream(c2rPlan.get(), stream);

    std::vector<float> hostSino;
    std::vector<float> hostOut(maxOut);
    std::vector<uint16_t> hostOut16(maxOut);
    qint64 fillMs = 0;
    float uploadMs = 0.0f;
    float fftMs = 0.0f;
    float bpMs = 0.0f;
    float downloadMs = 0.0f;
    qint64 enqueueMs = 0;
    const int totalChunks = (height + zChunk - 1) / zChunk;
    const std::vector<int> chunkStarts = makeCenterFirstChunkStarts(height, zChunk);
    std::unique_ptr<IChunkResultWriter> writer;
    OfflineSaveWindow saveWindow;
    int processedChunks = 0;
    for (int zStart : chunkStarts)
    {
        if (stopRequested.load())
        {
            break;
        }
        const int currH = std::min(zChunk, height - zStart);
        const int chunkIndex = zStart / zChunk;
        if (processedChunks == 0 || processedChunks == totalChunks - 1 || ((processedChunks + 1) % 10) == 0)
        {
            log(QStringLiteral("Offline FBP chunk %1/%2: z=%3 h=%4%5.")
                .arg(processedChunks + 1).arg(totalChunks).arg(zStart).arg(currH)
                .arg(processedChunks == 0 ? QStringLiteral(" (center calibration)") : QString()));
        }
        QElapsedTimer cpuTimer;
        cpuTimer.start();
        fillSinogramChunk(projectionStack, angleCount, height, width, zStart, currH, hostSino);
        fillMs += cpuTimer.elapsed();
        CudaEvent evStart(true);
        CudaEvent evAfterUpload(true);
        CudaEvent evAfterFft(true);
        CudaEvent evAfterBp(true);
        CudaEvent evAfterConvert(true);
        CudaEvent evAfterDownload(true);
        ensureCuda(cudaEventRecord(evStart.get(), stream), "record offline FBP start event");
        ensureCuda(cudaMemcpyAsync(deviceSino.ptr(), hostSino.data(),
            hostSino.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "upload offline FBP sinogram");
        ensureCuda(cudaEventRecord(evAfterUpload.get(), stream), "record offline FBP upload event");

        applyStripeRemovalFwIfEnabled(config, deviceSino.ptr(), currH, angleCount, width, stream, log);

        const int transformCount = currH * angleCount;
        if (!launchPrepareFbpInput(deviceSino.ptr(), devicePadded.ptr(), transformCount, width, paddedLength, stream))
        {
            throw std::runtime_error("Offline FBP prepare input failed.");
        }
        ensureCufft(cufftExecR2C(r2cPlan.get(),
            reinterpret_cast<cufftReal*>(devicePadded.ptr()), deviceSpectrum.ptr()), "offline FBP R2C");
        if (!launchMultiplyFbpSpectrum(deviceSpectrum.ptr(), deviceFilter.ptr(), transformCount, freqWidth, stream))
        {
            throw std::runtime_error("Offline FBP spectrum multiply failed.");
        }
        ensureCufft(cufftExecC2R(c2rPlan.get(), deviceSpectrum.ptr(),
            reinterpret_cast<cufftReal*>(devicePadded.ptr())), "offline FBP C2R");
        if (!launchExtractFbpOutput(devicePadded.ptr(), deviceSino.ptr(),
                transformCount, width, paddedLength, static_cast<float>(paddedLength), stream))
        {
            throw std::runtime_error("Offline FBP extract output failed.");
        }
        ensureCuda(cudaEventRecord(evAfterFft.get(), stream), "record offline FBP FFT event");
        launchOfflineFbpBackprojection(deviceSino.ptr(), deviceGeom, angleCount, width, currH,
            config.geometry.corShift, deviceOut.ptr(), stream);
        ensureCuda(cudaGetLastError(), "offline FBP backprojection launch");
        ensureCuda(cudaEventRecord(evAfterBp.get(), stream), "record offline FBP backprojection event");

        const size_t chunkVoxels = static_cast<size_t>(currH) * width * width;
        const bool needsCalibration = (writer == nullptr);
        if (needsCalibration)
        {
            ensureCuda(cudaMemcpyAsync(hostOut.data(), deviceOut.ptr(),
                chunkVoxels * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "download offline FBP calibration chunk");
            ensureCuda(cudaStreamSynchronize(stream), "offline FBP calibration sync");

            const std::vector<float> calibration(hostOut.begin(), hostOut.begin() + static_cast<std::ptrdiff_t>(chunkVoxels));
            saveWindow = makeOfflineSaveWindowFromCalibration(calibration);
            log(QStringLiteral("Offline FBP fixed uint16 save window: center chunk p%1-p%2 + %3% high margin, vmin=%4, vmax=%5.")
                .arg(kAutoSaveWindowLowPercentile, 0, 'f', 2)
                .arg(kAutoSaveWindowHighPercentile, 0, 'f', 2)
                .arg(kAutoSaveWindowHighMarginFraction * 100.0, 0, 'f', 1)
                .arg(saveWindow.vmin, 0, 'g', 8)
                .arg(saveWindow.vmax, 0, 'g', 8));
            writer = makeChunkWriter(config, saveWindow, log);
        }

        launchConvertFloatVolumeToUint16(deviceOut.ptr(), deviceOut16.ptr(), chunkVoxels,
            saveWindow.vmin, saveWindow.vmax, stream);
        ensureCuda(cudaGetLastError(), "offline FBP float-to-uint16 launch");
        ensureCuda(cudaEventRecord(evAfterConvert.get(), stream), "record offline FBP uint16 conversion event");
        ensureCuda(cudaMemcpyAsync(hostOut16.data(), deviceOut16.ptr(),
            chunkVoxels * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream),
            "download offline FBP uint16 chunk");
        ensureCuda(cudaEventRecord(evAfterDownload.get(), stream), "record offline FBP download event");
        ensureCuda(cudaStreamSynchronize(stream), "offline FBP chunk sync");
        float segmentMs = 0.0f;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evStart.get(), evAfterUpload.get()),
            "measure offline FBP upload");
        uploadMs += segmentMs;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evAfterUpload.get(), evAfterFft.get()),
            "measure offline FBP FFT");
        fftMs += segmentMs;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evAfterFft.get(), evAfterBp.get()),
            "measure offline FBP backprojection");
        bpMs += segmentMs;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evAfterConvert.get(), evAfterDownload.get()),
            "measure offline FBP download");
        downloadMs += segmentMs;
        cpuTimer.restart();

        ChunkSaveItem item;
        item.zStart = zStart;
        item.zCount = currH;
        item.width = width;
        item.data.assign(hostOut16.begin(), hostOut16.begin() + static_cast<std::ptrdiff_t>(chunkVoxels));
        writer->enqueue(std::move(item));
        enqueueMs += cpuTimer.elapsed();

        if (directPreview)
        {
            if (needsCalibration)
            {
                directPreview(makePreviewSliceFromChunk(hostOut, currH, width), processedChunks == 0);
            }
            else
            {
                directPreview(makePreviewSliceFromChunk(hostOut16, currH, width, saveWindow), false);
            }
        }
        ++processedChunks;
        progress(processedChunks, totalChunks);
    }

    const qint64 reconstructionMs = reconstructionWallTimer.elapsed();
    log(QStringLiteral("Offline FBP algorithm reconstruction complete in %1 s; waiting for save queue to finish.")
        .arg(formatSeconds(reconstructionMs)));

    if (writer)
    {
        writer->finish();
        const ChunkSaveStats stats = writer->stats();
        const double negativeRatio = stats.finiteSamples > 0
            ? static_cast<double>(stats.negativeClamped) * 100.0 / static_cast<double>(stats.finiteSamples)
            : 0.0;
        const double saturatedRatio = stats.finiteSamples > 0
            ? static_cast<double>(stats.saturated) * 100.0 / static_cast<double>(stats.finiteSamples)
            : 0.0;
        log(QStringLiteral("Offline FBP save stats: TIFF Slices, vmin=%1, vmax=%2, saved=%3 slices, wrote=%4 GiB, throughput=%5 MiB/s, saturated=%6%, zero-or-below=%7%, save-worker=%8s.")
            .arg(saveWindow.vmin, 0, 'g', 8)
            .arg(saveWindow.vmax, 0, 'g', 8)
            .arg(stats.savedSlices)
            .arg(formatGiB(static_cast<size_t>(stats.bytesWritten)))
            .arg(formatMiBPerSecond(stats.bytesWritten, stats.saveMs))
            .arg(saturatedRatio, 0, 'f', 4)
            .arg(negativeRatio, 0, 'f', 4)
            .arg(formatSeconds(stats.saveMs)));
    }

    log(QStringLiteral("Offline FBP timing: fill=%1s upload=%2s fft/filter=%3s backproject=%4s download=%5s enqueue=%6s.")
        .arg(formatSeconds(fillMs))
        .arg(QString::number(uploadMs / 1000.0f, 'f', 3))
        .arg(QString::number(fftMs / 1000.0f, 'f', 3))
        .arg(QString::number(bpMs / 1000.0f, 'f', 3))
        .arg(QString::number(downloadMs / 1000.0f, 'f', 3))
        .arg(formatSeconds(enqueueMs)));
    cudaFree(deviceGeom);
    cudaStreamDestroy(stream);
    return reconstructionMs;
}

qint64 runOfflineGridrec(const AppConfig& config, const std::vector<float>& projectionStack,
    std::atomic_bool& stopRequested,
    const OfflineTomoReconstruction::LogCallback& log,
    const OfflineTomoReconstruction::ProgressCallback& progress,
    const OfflineTomoReconstruction::DirectPreviewCallback& directPreview)
{
    QElapsedTimer reconstructionWallTimer;
    reconstructionWallTimer.start();
    const int angleCount = config.geometry.totalAngles;
    const int height = config.geometry.volH;
    const int width = config.geometry.volW;
    const int zChunk = std::min(config.compute.zChunkSize, height);
    const int paddedLength = paddedLengthForWidth(width);
    const int freqWidth = paddedLength / 2 + 1;
    const int gridSize = width;
    const int cropStart = 0;

    DeviceFloatBuffer deviceSino;
    DeviceFloatBuffer devicePadded;
    DeviceComplexBuffer deviceSpectrum;
    DeviceFloatBuffer deviceFilter;
    DeviceComplexBuffer deviceGrid;
    DeviceComplexBuffer deviceWork;
    DeviceFloatBuffer deviceOut;
    DeviceUint16Buffer deviceOut16;
    DeviceFloatBuffer deviceDeapod;
    double* deviceVectors = nullptr;
    cudaStream_t stream = nullptr;
    CufftHandle r2cPlan;
    CufftHandle ifftPlan;

    const size_t maxSino = static_cast<size_t>(zChunk) * angleCount * width;
    const size_t maxPadded = static_cast<size_t>(zChunk) * angleCount * paddedLength;
    const size_t maxSpectrum = static_cast<size_t>(zChunk) * angleCount * freqWidth;
    const size_t gridElements = static_cast<size_t>(zChunk) * gridSize * gridSize;
    const size_t maxOut = static_cast<size_t>(zChunk) * width * width;

    if (!deviceSino.allocate(maxSino) || !devicePadded.allocate(maxPadded)
        || !deviceSpectrum.allocate(maxSpectrum) || !deviceFilter.allocate(freqWidth)
        || !deviceGrid.allocate(gridElements) || !deviceWork.allocate(gridElements)
        || !deviceOut.allocate(maxOut) || !deviceOut16.allocate(maxOut)
        || !deviceDeapod.allocate(static_cast<size_t>(gridSize) * gridSize))
    {
        throw std::runtime_error("Failed to allocate offline Gridrec GPU buffers.");
    }

    const auto filter = makeFbpFilter(paddedLength, config.preprocess.filterType);
    ensureCuda(cudaMemcpy(deviceFilter.ptr(), filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice),
        "upload offline Gridrec filter");
    const auto deapod = makeGridrecDeapodMap(gridSize);
    ensureCuda(cudaMemcpy(deviceDeapod.ptr(), deapod.data(), deapod.size() * sizeof(float), cudaMemcpyHostToDevice),
        "upload offline Gridrec deapodization map");
    const auto vectors = makeGridrecVectors(angleCount, config.geometry.corShift, config.geometry.angleOffsetDeg);
    ensureCuda(cudaMalloc(&deviceVectors, vectors.size() * sizeof(double)), "allocate offline Gridrec vectors");
    ensureCuda(cudaMemcpy(deviceVectors, vectors.data(), vectors.size() * sizeof(double), cudaMemcpyHostToDevice),
        "upload offline Gridrec vectors");
    ensureCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create offline Gridrec stream");

    int rfftN[1] = { paddedLength };
    int rfftInembed[1] = { paddedLength };
    int rfftOnembed[1] = { freqWidth };
    const int maxTransformCount = zChunk * angleCount;
    ensureCufft(cufftPlanMany(r2cPlan.out(), 1, rfftN,
        rfftInembed, 1, paddedLength,
        rfftOnembed, 1, freqWidth,
        CUFFT_R2C, maxTransformCount), "offline Gridrec R2C plan");
    cufftSetStream(r2cPlan.get(), stream);

    int ifftN[2] = { gridSize, gridSize };
    int ifftEmbed[2] = { gridSize, gridSize };
    ensureCufft(cufftPlanMany(ifftPlan.out(), 2, ifftN,
        ifftEmbed, 1, gridSize * gridSize,
        ifftEmbed, 1, gridSize * gridSize,
        CUFFT_C2C, zChunk), "offline Gridrec IFFT plan");
    cufftSetStream(ifftPlan.get(), stream);

    std::vector<float> hostSino;
    std::vector<float> hostOut(maxOut);
    std::vector<uint16_t> hostOut16(maxOut);
    qint64 fillMs = 0;
    float uploadMs = 0.0f;
    float fftMs = 0.0f;
    float accumulateMs = 0.0f;
    float finalizeMs = 0.0f;
    float downloadMs = 0.0f;
    qint64 enqueueMs = 0;
    const int totalChunks = (height + zChunk - 1) / zChunk;
    const std::vector<int> chunkStarts = makeCenterFirstChunkStarts(height, zChunk);
    std::unique_ptr<IChunkResultWriter> writer;
    OfflineSaveWindow saveWindow;
    int processedChunks = 0;
    for (int zStart : chunkStarts)
    {
        if (stopRequested.load())
        {
            break;
        }
        const int currH = std::min(zChunk, height - zStart);
        if (processedChunks == 0 || processedChunks == totalChunks - 1 || ((processedChunks + 1) % 10) == 0)
        {
            log(QStringLiteral("Offline Gridrec chunk %1/%2: z=%3 h=%4%5.")
                .arg(processedChunks + 1).arg(totalChunks).arg(zStart).arg(currH)
                .arg(processedChunks == 0 ? QStringLiteral(" (center calibration)") : QString()));
        }
        QElapsedTimer cpuTimer;
        cpuTimer.start();
        fillSinogramChunk(projectionStack, angleCount, height, width, zStart, currH, hostSino);
        fillMs += cpuTimer.elapsed();
        CudaEvent evStart(true);
        CudaEvent evAfterUpload(true);
        CudaEvent evAfterFft(true);
        CudaEvent evAfterAccumulate(true);
        CudaEvent evAfterFinalize(true);
        CudaEvent evAfterConvert(true);
        CudaEvent evAfterDownload(true);
        ensureCuda(cudaEventRecord(evStart.get(), stream), "record offline Gridrec start event");
        ensureCuda(cudaMemcpyAsync(deviceSino.ptr(), hostSino.data(),
            hostSino.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "upload offline Gridrec sinogram");
        ensureCuda(cudaEventRecord(evAfterUpload.get(), stream), "record offline Gridrec upload event");

        applyStripeRemovalFwIfEnabled(config, deviceSino.ptr(), currH, angleCount, width, stream, log);

        const int transformCount = currH * angleCount;
        if (!launchPrepareFbpInput(deviceSino.ptr(), devicePadded.ptr(), transformCount, width, paddedLength, stream))
        {
            throw std::runtime_error("Offline Gridrec prepare FFT input failed.");
        }
        ensureCufft(cufftExecR2C(r2cPlan.get(),
            reinterpret_cast<cufftReal*>(devicePadded.ptr()), deviceSpectrum.ptr()), "offline Gridrec R2C");
        if (!launchMultiplyFbpSpectrum(deviceSpectrum.ptr(), deviceFilter.ptr(), transformCount, freqWidth, stream))
        {
            throw std::runtime_error("Offline Gridrec filter multiply failed.");
        }
        ensureCuda(cudaEventRecord(evAfterFft.get(), stream), "record offline Gridrec FFT event");

        ensureCuda(cudaMemsetAsync(deviceGrid.ptr(), 0, gridElements * sizeof(cufftComplex), stream),
            "clear offline Gridrec accumulator");
        launchGridrecAccumulate(deviceSpectrum.ptr(), deviceVectors, angleCount, currH, freqWidth,
            gridSize, paddedLength, deviceGrid.ptr(), stream);
        ensureCuda(cudaGetLastError(), "offline Gridrec accumulate launch");
        ensureCuda(cudaEventRecord(evAfterAccumulate.get(), stream), "record offline Gridrec accumulate event");
        launchGridrecIfftshiftCopy(deviceGrid.ptr(), currH, gridSize, gridSize, deviceWork.ptr(), stream);
        ensureCuda(cudaGetLastError(), "offline Gridrec ifftshift launch");
        ensureCufft(cufftExecC2C(ifftPlan.get(), deviceWork.ptr(), deviceWork.ptr(), CUFFT_INVERSE),
            "offline Gridrec IFFT");
        launchGridrecFinalizeCrop(deviceWork.ptr(), deviceDeapod.ptr(), currH, gridSize, gridSize,
            width, width, cropStart, static_cast<float>(gridSize) * 4.0f, deviceOut.ptr(), stream);
        ensureCuda(cudaGetLastError(), "offline Gridrec finalize launch");
        ensureCuda(cudaEventRecord(evAfterFinalize.get(), stream), "record offline Gridrec finalize event");

        const size_t chunkVoxels = static_cast<size_t>(currH) * width * width;
        const bool needsCalibration = (writer == nullptr);
        if (needsCalibration)
        {
            ensureCuda(cudaMemcpyAsync(hostOut.data(), deviceOut.ptr(),
                chunkVoxels * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "download offline Gridrec calibration chunk");
            ensureCuda(cudaStreamSynchronize(stream), "offline Gridrec calibration sync");

            const std::vector<float> calibration(hostOut.begin(), hostOut.begin() + static_cast<std::ptrdiff_t>(chunkVoxels));
            saveWindow = makeOfflineSaveWindowFromCalibration(calibration);
            log(QStringLiteral("Offline Gridrec fixed uint16 save window: center chunk p%1-p%2 + %3% high margin, vmin=%4, vmax=%5.")
                .arg(kAutoSaveWindowLowPercentile, 0, 'f', 2)
                .arg(kAutoSaveWindowHighPercentile, 0, 'f', 2)
                .arg(kAutoSaveWindowHighMarginFraction * 100.0, 0, 'f', 1)
                .arg(saveWindow.vmin, 0, 'g', 8)
                .arg(saveWindow.vmax, 0, 'g', 8));
            writer = makeChunkWriter(config, saveWindow, log);
        }

        launchConvertFloatVolumeToUint16(deviceOut.ptr(), deviceOut16.ptr(), chunkVoxels,
            saveWindow.vmin, saveWindow.vmax, stream);
        ensureCuda(cudaGetLastError(), "offline Gridrec float-to-uint16 launch");
        ensureCuda(cudaEventRecord(evAfterConvert.get(), stream), "record offline Gridrec uint16 conversion event");
        ensureCuda(cudaMemcpyAsync(hostOut16.data(), deviceOut16.ptr(),
            chunkVoxels * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream),
            "download offline Gridrec uint16 chunk");
        ensureCuda(cudaEventRecord(evAfterDownload.get(), stream), "record offline Gridrec download event");
        ensureCuda(cudaStreamSynchronize(stream), "offline Gridrec chunk sync");
        float segmentMs = 0.0f;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evStart.get(), evAfterUpload.get()),
            "measure offline Gridrec upload");
        uploadMs += segmentMs;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evAfterUpload.get(), evAfterFft.get()),
            "measure offline Gridrec FFT");
        fftMs += segmentMs;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evAfterFft.get(), evAfterAccumulate.get()),
            "measure offline Gridrec accumulate");
        accumulateMs += segmentMs;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evAfterAccumulate.get(), evAfterFinalize.get()),
            "measure offline Gridrec finalize");
        finalizeMs += segmentMs;
        ensureCuda(cudaEventElapsedTime(&segmentMs, evAfterConvert.get(), evAfterDownload.get()),
            "measure offline Gridrec download");
        downloadMs += segmentMs;
        cpuTimer.restart();

        ChunkSaveItem item;
        item.zStart = zStart;
        item.zCount = currH;
        item.width = width;
        item.data.assign(hostOut16.begin(), hostOut16.begin() + static_cast<std::ptrdiff_t>(chunkVoxels));
        writer->enqueue(std::move(item));
        enqueueMs += cpuTimer.elapsed();

        if (directPreview)
        {
            if (needsCalibration)
            {
                directPreview(makePreviewSliceFromChunk(hostOut, currH, width), processedChunks == 0);
            }
            else
            {
                directPreview(makePreviewSliceFromChunk(hostOut16, currH, width, saveWindow), false);
            }
        }
        ++processedChunks;
        progress(processedChunks, totalChunks);
    }

    const qint64 reconstructionMs = reconstructionWallTimer.elapsed();
    log(QStringLiteral("Offline Gridrec algorithm reconstruction complete in %1 s; waiting for save queue to finish.")
        .arg(formatSeconds(reconstructionMs)));

    if (writer)
    {
        writer->finish();
        const ChunkSaveStats stats = writer->stats();
        const double negativeRatio = stats.finiteSamples > 0
            ? static_cast<double>(stats.negativeClamped) * 100.0 / static_cast<double>(stats.finiteSamples)
            : 0.0;
        const double saturatedRatio = stats.finiteSamples > 0
            ? static_cast<double>(stats.saturated) * 100.0 / static_cast<double>(stats.finiteSamples)
            : 0.0;
        log(QStringLiteral("Offline Gridrec save stats: TIFF Slices, vmin=%1, vmax=%2, saved=%3 slices, wrote=%4 GiB, throughput=%5 MiB/s, saturated=%6%, zero-or-below=%7%, save-worker=%8s.")
            .arg(saveWindow.vmin, 0, 'g', 8)
            .arg(saveWindow.vmax, 0, 'g', 8)
            .arg(stats.savedSlices)
            .arg(formatGiB(static_cast<size_t>(stats.bytesWritten)))
            .arg(formatMiBPerSecond(stats.bytesWritten, stats.saveMs))
            .arg(saturatedRatio, 0, 'f', 4)
            .arg(negativeRatio, 0, 'f', 4)
            .arg(formatSeconds(stats.saveMs)));
    }

    log(QStringLiteral("Offline Gridrec timing: fill=%1s upload=%2s fft/filter=%3s accumulate=%4s finalize=%5s download=%6s enqueue=%7s.")
        .arg(formatSeconds(fillMs))
        .arg(QString::number(uploadMs / 1000.0f, 'f', 3))
        .arg(QString::number(fftMs / 1000.0f, 'f', 3))
        .arg(QString::number(accumulateMs / 1000.0f, 'f', 3))
        .arg(QString::number(finalizeMs / 1000.0f, 'f', 3))
        .arg(QString::number(downloadMs / 1000.0f, 'f', 3))
        .arg(formatSeconds(enqueueMs)));
    cudaFree(deviceVectors);
    cudaStreamDestroy(stream);
    return reconstructionMs;
}

} // namespace

OfflineTomoResult OfflineTomoReconstruction::run(const AppConfig& inputConfig,
    std::atomic_bool& stopRequested,
    const LogCallback& log,
    const ProgressCallback& progress,
    const CorCallback& corComputed,
    const GeometryCallback& geometryResolved,
    const PreviewCallback& previewSuggested,
    const DirectPreviewCallback& directPreview)
{
    (void)previewSuggested;
    AppConfig config = inputConfig;
    appendReconDebug(QStringLiteral("OfflineTomoReconstruction run enter method=%1 stripe=%2 level=%3 watch=%4 save=%5")
        .arg(toString(config.io.reconMethod))
        .arg(config.stripe.enable)
        .arg(config.stripe.level)
        .arg(config.io.watchDir)
        .arg(config.io.saveDir));
    config.io.tiffThreads = std::clamp(config.io.tiffThreads, 1, 64);
    if (config.io.watchDir.isEmpty())
    {
        throw std::runtime_error("WATCH_DIR is empty.");
    }
    if (config.compute.batchSize <= 0 || config.compute.zChunkSize <= 0)
    {
        throw std::runtime_error("BATCH_SIZE and Z_CHUNK_SIZE must be > 0.");
    }
    ensureCuda(cudaSetDevice(config.compute.gpuIndex), "select CUDA device");

    QElapsedTimer totalTimer;
    totalTimer.start();
    FileProjectionSource source(config.io);
    cv::Mat flat;
    cv::Mat corFlat180;
    cv::Mat dark;

    log(QStringLiteral("=== Starting %1 ===").arg(toString(config.io.reconMethod)));
    log(QStringLiteral("Offline TIFF writer threads requested: %1.").arg(config.io.tiffThreads));
    if (config.stripe.enable)
    {
        log(QStringLiteral("FW stripe removal enabled: sigma=%1, level=%2, wname=%3, z-chunk=%4. Runs on sinograms before Offline FBP/Gridrec filtering.")
            .arg(config.stripe.sigma, 0, 'g', 6)
            .arg(config.stripe.level)
            .arg(config.stripe.waveletName.isEmpty() ? QStringLiteral("db5") : config.stripe.waveletName)
            .arg(std::max(1, config.stripe.chunk)));
        if (config.stripe.level > 0)
        {
            log(QStringLiteral("FW stripe uses db5 DWT/IDWT and damps vertical detail coefficients in Fourier space."));
        }
    }
    if (config.io.correctionMode == CorrectionMode::FlatDark)
    {
        log(QStringLiteral("Offline waiting for dark reference..."));
        if (!source.waitForReferenceImage(config.io.darkPrefix, QStringLiteral("Dark"), dark, stopRequested))
        {
            throw std::runtime_error("Failed to acquire dark reference.");
        }
    }

    if (config.io.correctionMode == CorrectionMode::Flat || config.io.correctionMode == CorrectionMode::FlatDark)
    {
        log(QStringLiteral("Offline loading flat groups for reference geometry/COR..."));
        const std::vector<FlatGroupConfig> flatGroupConfigs = flatGroupConfigsForRun(source, config, stopRequested, log);
        if (flatGroupConfigs.empty())
        {
            throw std::runtime_error("Flat correction requested but no flat groups were resolved.");
        }
        flat = averageFlatGroup(source, config, flatGroupConfigs.front(), stopRequested);
        corFlat180 = averageFlatGroup(source, config, flatGroupConfigs.back(), stopRequested);
    }

    cv::Mat ref0;
    cv::Mat ref180;
    QString ref0Path;
    QString ref180Path;
    if (config.geometry.corAuto)
    {
        log(QStringLiteral("Offline waiting for 0 deg and 180 deg reference files..."));
        if (!source.waitForSpecificFile(config.io.filePattern0, QStringLiteral("0 Deg"), ref0, &ref0Path, stopRequested)
            || !source.waitForSpecificFile(config.io.filePattern180, QStringLiteral("180 Deg"), ref180, &ref180Path, stopRequested))
        {
            throw std::runtime_error("Failed to acquire COR reference files.");
        }
    }
    else if (flat.empty() && (config.geometry.volW <= 0 || config.geometry.volH <= 0))
    {
        log(QStringLiteral("Offline Auto COR disabled; loading first projection to resolve geometry."));
        if (!source.waitForSpecificFile(config.io.filePattern0, QStringLiteral("0 Deg"), ref0, &ref0Path, stopRequested))
        {
            throw std::runtime_error("Failed to acquire first projection for geometry.");
        }
    }

    resolveOfflineGeometryFromConfigOrReferences(config, flat, dark, ref0, ref180, log);

    const size_t projectionStackBytes = static_cast<size_t>(config.geometry.totalAngles)
        * config.geometry.volH * config.geometry.volW * sizeof(float);
    const size_t chunkBytes = static_cast<size_t>(std::min(config.compute.zChunkSize, config.geometry.volH))
        * config.geometry.volW * config.geometry.volW * sizeof(float);
    const size_t saveChunkBytes = static_cast<size_t>(std::min(config.compute.zChunkSize, config.geometry.volH))
        * config.geometry.volW * config.geometry.volW * sizeof(uint16_t);
    log(QStringLiteral("Offline memory estimate: preprocessed projection stack=%1 GiB, calibration float chunk=%2 GiB, TIFF uint16 chunk=%3 GiB; full CPU volume is not allocated.")
        .arg(formatGiB(projectionStackBytes), formatGiB(chunkBytes), formatGiB(saveChunkBytes)));
    if (config.geometry.corAuto)
    {
        log(QStringLiteral("Offline estimating COR..."));
        CorEstimatorFft corEstimator;
        CorrectionContext context;
        context.mode = config.io.correctionMode;
        context.flat = flat;
        context.dark = dark;
        context.cropMargin = 50;
        CorrectionContext context180 = context;
        context180.flat = corFlat180.empty() ? flat : corFlat180;
        config.geometry.corShift = corEstimator.estimateShift(ref0, context, ref180, context180);
        corComputed(config.geometry.corShift);
        log(QStringLiteral("Offline Auto COR Result: %1").arg(config.geometry.corShift, 0, 'f', 4));
    }
    geometryResolved(config.geometry.volW, config.geometry.volH);

    QElapsedTimer stageTimer;
    stageTimer.start();
    std::vector<float> projectionStack;
    preprocessAllProjections(config, source, flat, dark, projectionStack, stopRequested, log, progress);
    if (stopRequested.load())
    {
        return OfflineTomoResult{ nullptr, config };
    }
    log(QStringLiteral("Offline preprocess complete in %1 s.").arg(formatSeconds(stageTimer.elapsed())));

    stageTimer.restart();
    qint64 algorithmReconstructionMs = 0;
    if (config.io.reconMethod == ReconMethod::OfflineFBP)
    {
        algorithmReconstructionMs = runOfflineFbp(config, projectionStack, stopRequested, log, progress, directPreview);
    }
    else if (config.io.reconMethod == ReconMethod::OfflineGridrec)
    {
        algorithmReconstructionMs = runOfflineGridrec(config, projectionStack, stopRequested, log, progress, directPreview);
    }
    else
    {
        throw std::runtime_error("OfflineTomoReconstruction called with non-offline method.");
    }
    if (!stopRequested.load())
    {
        log(QStringLiteral("%1 TIFF slices saved during reconstruction to %2.")
            .arg(toString(config.io.reconMethod), QDir::toNativeSeparators(config.io.saveDir)));
    }
    log(QStringLiteral("%1 algorithm reconstruction time, excluding save queue drain: %2 s.")
        .arg(toString(config.io.reconMethod), formatSeconds(algorithmReconstructionMs)));
    log(QStringLiteral("%1 reconstruction stage including save queue drain: %2 s.")
        .arg(toString(config.io.reconMethod), formatSeconds(stageTimer.elapsed())));
    log(QStringLiteral("%1 finished including preprocess and save queue drain: %2 s.")
        .arg(toString(config.io.reconMethod), formatSeconds(totalTimer.elapsed())));
    return OfflineTomoResult{ nullptr, config };
}

} // namespace ct
