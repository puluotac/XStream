#include "core/recon/OfflineTomoReconstruction.h"

#include "core/cor/CorEstimatorFft.h"
#include "core/io/FileProjectionSource.h"
#include "core/preprocess/ImageNormalizer.h"
#include "core/preprocess/PaganinProcessor.h"
#include "core/recon/GpuProjectionBatch.h"
#include "cuda/offline_tomo_kernels.h"
#include "cuda/preprocess_kernels.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QThread>

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

constexpr double kAutoSaveWindowLowPercentile = 1.0;
constexpr double kAutoSaveWindowHighPercentile = 99.9;

struct OfflineInputBatch
{
    int batchIndex = 0;
    int validCount = 0;
    std::vector<int> logicalIndices;
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

void preprocessAllProjections(AppConfig& config, FileProjectionSource& source, const cv::Mat& flat, const cv::Mat& dark,
    std::vector<float>& projectionStack, std::atomic_bool& stopRequested,
    const OfflineTomoReconstruction::LogCallback& log,
    const OfflineTomoReconstruction::ProgressCallback& progress)
{
    const int angleCount = config.geometry.totalAngles;
    const int height = config.geometry.volH;
    const int width = config.geometry.volW;
    const size_t projectionElements = static_cast<size_t>(angleCount) * height * width;
    projectionStack.assign(projectionElements, 0.0f);

    HostPinnedProjectionBatch hostBatches[2];
    GpuProjectionBatch gpuBatch;
    DeviceCorrectionImages correctionImages;
    PaganinProcessor paganin;
    cudaStream_t stream = nullptr;

    if (!hostBatches[0].initialize(config.compute.batchSize, height, width)
        || !hostBatches[1].initialize(config.compute.batchSize, height, width)
        || !gpuBatch.initialize(config.compute.batchSize, height, width))
    {
        throw std::runtime_error("Failed to allocate offline preprocessing batch buffers.");
    }
    if (!ImageNormalizer::uploadCorrectionImages(flat, dark, height, width, correctionImages))
    {
        throw std::runtime_error("Failed to upload offline correction images.");
    }
    paganin.configure(config.phase);
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
        if (!ImageNormalizer::applyCorrection(gpuBatch, correctionImages, config.io.correctionMode, stream))
        {
            throw std::runtime_error("Offline flat/dark correction failed.");
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
    correctionImages.release();

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

            std::vector<float> lowCalibration(hostOut.begin(), hostOut.begin() + static_cast<std::ptrdiff_t>(chunkVoxels));
            std::vector<float> highCalibration = lowCalibration;
            saveWindow.vmin = percentileFinite(lowCalibration, kAutoSaveWindowLowPercentile);
            saveWindow.vmax = percentileFinite(highCalibration, kAutoSaveWindowHighPercentile);
            if (saveWindow.vmax <= saveWindow.vmin)
            {
                saveWindow.vmax = saveWindow.vmin + 1.0f;
            }
            log(QStringLiteral("Offline FBP fixed uint16 save window: center chunk p%1-p%2, vmin=%3, vmax=%4.")
                .arg(kAutoSaveWindowLowPercentile, 0, 'f', 2)
                .arg(kAutoSaveWindowHighPercentile, 0, 'f', 2)
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

            std::vector<float> lowCalibration(hostOut.begin(), hostOut.begin() + static_cast<std::ptrdiff_t>(chunkVoxels));
            std::vector<float> highCalibration = lowCalibration;
            saveWindow.vmin = percentileFinite(lowCalibration, kAutoSaveWindowLowPercentile);
            saveWindow.vmax = percentileFinite(highCalibration, kAutoSaveWindowHighPercentile);
            if (saveWindow.vmax <= saveWindow.vmin)
            {
                saveWindow.vmax = saveWindow.vmin + 1.0f;
            }
            log(QStringLiteral("Offline Gridrec fixed uint16 save window: center chunk p%1-p%2, vmin=%3, vmax=%4.")
                .arg(kAutoSaveWindowLowPercentile, 0, 'f', 2)
                .arg(kAutoSaveWindowHighPercentile, 0, 'f', 2)
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
    config.io.tiffThreads = std::clamp(config.io.tiffThreads, 1, 64);
    if (config.phase.enable)
    {
        config.phase.enable = false;
        log(QStringLiteral("Offline FBP/Gridrec ignores PHASE.ENABLE: use camera-side phase retrieval and point WATCH_DIR to processed projections."));
    }
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
    cv::Mat dark;

    log(QStringLiteral("=== Starting %1 ===").arg(toString(config.io.reconMethod)));
    log(QStringLiteral("Offline TIFF writer threads requested: %1.").arg(config.io.tiffThreads));
    if (config.io.correctionMode == CorrectionMode::Flat || config.io.correctionMode == CorrectionMode::FlatDark)
    {
        log(QStringLiteral("Offline waiting for flat reference..."));
        if (!source.waitForReferenceImage(config.io.flatPrefix, QStringLiteral("Flat"), flat, stopRequested))
        {
            throw std::runtime_error("Failed to acquire flat reference.");
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

    cv::Mat ref0;
    cv::Mat ref180;
    QString ref0Path;
    QString ref180Path;
    log(QStringLiteral("Offline waiting for 0 deg and 180 deg reference files..."));
    if (!source.waitForSpecificFile(config.io.filePattern0, QStringLiteral("0 Deg"), ref0, &ref0Path, stopRequested)
        || !source.waitForSpecificFile(config.io.filePattern180, QStringLiteral("180 Deg"), ref180, &ref180Path, stopRequested))
    {
        throw std::runtime_error("Failed to acquire COR reference files.");
    }

    config.geometry.volH = !flat.empty() ? flat.rows : ref0.rows;
    config.geometry.volW = !flat.empty() ? flat.cols : ref0.cols;
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
        config.geometry.corShift = corEstimator.estimateShift(ref0, ref180, context);
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
