#include "core/recon/ReconstructionWorker.h"

#include "core/common/GeometryVectorGenerator.h"
#include "core/common/ProjectionTypes.h"
#include "core/cor/CorEstimatorFft.h"
#include "core/io/FileProjectionSource.h"
#include "core/preprocess/ImageNormalizer.h"
#include "core/recon/Bp3dCudaBackend.h"
#include "core/recon/CpuVolumeBuffer.h"
#include "core/recon/GpuVolumeBuffer.h"
#include "core/recon/IReconstructionVolume.h"
#include "core/recon/OfflineTomoReconstruction.h"
#include "core/recon/ReconstructionRuntimeCache.h"

#include <QElapsedTimer>
#include <QCoreApplication>
#include <QDateTime>
#include <QFile>
#include <QThread>
#include <QTextStream>

#include <cuda_runtime.h>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ct
{

namespace
{

constexpr int kLivePreviewEveryNBatches = 4;
constexpr qint64 kLivePreviewMinIntervalMs = 250;
constexpr size_t kMaxPinnedSlabBytes = static_cast<size_t>(1024) * 1024 * 1024;

QString formatGiB(size_t bytes)
{
    return QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 'f', 2);
}

void appendReconDebug(const QString& text)
{
    QFile file(QCoreApplication::applicationDirPath() + QStringLiteral("/ct_recon_debug.log"));
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        QTextStream stream(&file);
        stream << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << " " << text << '\n';
    }
}

size_t volumeBytes(int depth, int height, int width)
{
    if (depth <= 0 || height <= 0 || width <= 0)
    {
        return 0;
    }
    const size_t d = static_cast<size_t>(depth);
    const size_t h = static_cast<size_t>(height);
    const size_t w = static_cast<size_t>(width);
    if (d > std::numeric_limits<size_t>::max() / h || d * h > std::numeric_limits<size_t>::max() / w
        || d * h * w > std::numeric_limits<size_t>::max() / sizeof(float))
    {
        return std::numeric_limits<size_t>::max();
    }
    return d * h * w * sizeof(float);
}

size_t availablePhysicalMemoryBytes()
{
#ifdef Q_OS_WIN
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
    {
        return static_cast<size_t>(status.ullAvailPhys);
    }
#endif
    return std::numeric_limits<size_t>::max();
}

std::vector<double> sliceVectors(const std::vector<double>& allVectors, const std::vector<int>& logicalIndices)
{
    std::vector<double> result;
    result.reserve(static_cast<size_t>(logicalIndices.size()) * 12);

    for (const int index : logicalIndices)
    {
        const size_t base = static_cast<size_t>(index) * 12;
        result.insert(result.end(), allVectors.begin() + static_cast<std::ptrdiff_t>(base),
            allVectors.begin() + static_cast<std::ptrdiff_t>(base + 12));
    }

    return result;
}

class PinnedFloatBuffer
{
public:
    PinnedFloatBuffer() = default;
    ~PinnedFloatBuffer()
    {
        release();
    }

    PinnedFloatBuffer(const PinnedFloatBuffer&) = delete;
    PinnedFloatBuffer& operator=(const PinnedFloatBuffer&) = delete;

    bool allocate(size_t elementCount)
    {
        release();
        if (elementCount == 0)
        {
            return false;
        }

        if (cudaMallocHost(&data_, elementCount * sizeof(float)) != cudaSuccess)
        {
            data_ = nullptr;
            capacity_ = 0;
            return false;
        }

        capacity_ = elementCount;
        return true;
    }

    void release()
    {
        if (data_ != nullptr)
        {
            cudaFreeHost(data_);
            data_ = nullptr;
        }
        capacity_ = 0;
    }

    float* data()
    {
        return data_;
    }

    const float* data() const
    {
        return data_;
    }

    size_t capacity() const
    {
        return capacity_;
    }

private:
    float* data_ = nullptr;
    size_t capacity_ = 0;
};

class SlabPipelineResources
{
public:
    SlabPipelineResources() = default;
    ~SlabPipelineResources()
    {
        release();
    }

    SlabPipelineResources(const SlabPipelineResources&) = delete;
    SlabPipelineResources& operator=(const SlabPipelineResources&) = delete;

    bool initialize()
    {
        release();
        if (cudaStreamCreateWithFlags(&uploadStream_, cudaStreamNonBlocking) != cudaSuccess
            || cudaStreamCreateWithFlags(&downloadStream_, cudaStreamNonBlocking) != cudaSuccess)
        {
            release();
            return false;
        }

        for (int i = 0; i < 2; ++i)
        {
            if (cudaEventCreateWithFlags(&uploadDone_[i], cudaEventDisableTiming) != cudaSuccess
                || cudaEventCreateWithFlags(&reconDone_[i], cudaEventDisableTiming) != cudaSuccess
                || cudaEventCreateWithFlags(&downloadDone_[i], cudaEventDisableTiming) != cudaSuccess)
            {
                release();
                return false;
            }
        }

        return true;
    }

    void release()
    {
        for (int i = 0; i < 2; ++i)
        {
            if (uploadDone_[i] != nullptr)
            {
                cudaEventDestroy(uploadDone_[i]);
                uploadDone_[i] = nullptr;
            }
            if (reconDone_[i] != nullptr)
            {
                cudaEventDestroy(reconDone_[i]);
                reconDone_[i] = nullptr;
            }
            if (downloadDone_[i] != nullptr)
            {
                cudaEventDestroy(downloadDone_[i]);
                downloadDone_[i] = nullptr;
            }
        }

        if (uploadStream_ != nullptr)
        {
            cudaStreamDestroy(uploadStream_);
            uploadStream_ = nullptr;
        }
        if (downloadStream_ != nullptr)
        {
            cudaStreamDestroy(downloadStream_);
            downloadStream_ = nullptr;
        }
    }

    cudaStream_t uploadStream() const
    {
        return uploadStream_;
    }

    cudaStream_t downloadStream() const
    {
        return downloadStream_;
    }

    cudaEvent_t uploadDone(int slot) const
    {
        return uploadDone_[slot];
    }

    cudaEvent_t reconDone(int slot) const
    {
        return reconDone_[slot];
    }

    cudaEvent_t downloadDone(int slot) const
    {
        return downloadDone_[slot];
    }

private:
    cudaStream_t uploadStream_ = nullptr;
    cudaStream_t downloadStream_ = nullptr;
    cudaEvent_t uploadDone_[2] = { nullptr, nullptr };
    cudaEvent_t reconDone_[2] = { nullptr, nullptr };
    cudaEvent_t downloadDone_[2] = { nullptr, nullptr };
};

} // namespace

ReconstructionWorker::ReconstructionWorker(
    AppConfig config,
    std::shared_ptr<ReconstructionRuntimeCache> runtimeCache,
    QObject* parent)
    : QObject(parent)
    , config_(std::move(config))
    , runtimeCache_(std::move(runtimeCache))
{
    previewSliceIndex_.store(config_.preview.liveSliceIndex);
}

std::shared_ptr<IReconstructionVolume> ReconstructionWorker::volumeBuffer() const
{
    return volumeBuffer_;
}

AppConfig ReconstructionWorker::config() const
{
    return config_;
}

void ReconstructionWorker::setPreviewSliceIndex(int index)
{
    previewSliceIndex_.store(index);
    config_.preview.liveSliceIndex = index;
}

void ReconstructionWorker::requestStop()
{
    stopRequested_.store(true);
}

void ReconstructionWorker::process()
{
    try
    {
        appendReconDebug(QStringLiteral("worker process enter method=%1 stripe=%2 level=%3")
            .arg(toString(config_.io.reconMethod))
            .arg(config_.stripe.enable)
            .arg(config_.stripe.level));
        const auto finishStopped = [this]() {
            emit logMessage(QStringLiteral(">>> Reconstruction stopped."));
            emit reconstructionFinished();
        };

        stopRequested_.store(false);
        emit logMessage(QStringLiteral("=== Starting C++ Custom Recon ==="));
        emit logMessage(QStringLiteral("Recon method dispatch: %1.").arg(toString(config_.io.reconMethod)));
        appendReconDebug(QStringLiteral("worker emitted start logs"));

        if (config_.io.reconMethod == ReconMethod::OfflineFBP
            || config_.io.reconMethod == ReconMethod::OfflineGridrec)
        {
            appendReconDebug(QStringLiteral("worker entering OfflineTomoReconstruction"));
            auto result = OfflineTomoReconstruction::run(config_, stopRequested_,
                [this](const QString& text) { emit logMessage(text); },
                [this](int current, int total) { emit progressUpdated(current, total); },
                [this](double shift) { emit corComputed(shift); },
                [this](int volW, int volH) { emit volumeGeometryResolved(volW, volH); },
                [this](bool updateHistogram) { emit previewRefreshSuggested(updateHistogram); },
                [this](const cv::Mat& slice, bool updateHistogram) {
                    emit directPreviewSliceReady(slice.clone(), updateHistogram);
                });
            volumeBuffer_ = result.volume;
            config_ = result.config;
            appendReconDebug(QStringLiteral("worker OfflineTomoReconstruction returned"));
            if (stopRequested_.load())
            {
                finishStopped();
                return;
            }
            emit reconstructionFinished();
            return;
        }

        if (runtimeCache_ == nullptr)
        {
            throw std::runtime_error("Runtime cache is unavailable.");
        }

        if (config_.compute.batchSize <= 0)
        {
            throw std::runtime_error("BATCH_SIZE must be > 0.");
        }
        if (config_.compute.zChunkSize <= 0)
        {
            throw std::runtime_error("Z_CHUNK_SIZE must be > 0.");
        }
        if (cudaSetDevice(config_.compute.gpuIndex) != cudaSuccess)
        {
            throw std::runtime_error("Failed to select CUDA device.");
        }

        const bool previewEnabledDuringRun = config_.preview.enableLive;
        if (!previewEnabledDuringRun)
        {
            emit logMessage(QStringLiteral("Live preview disabled: rendering suspended during reconstruction."));
        }

        if (config_.io.watchDir.isEmpty())
        {
            throw std::runtime_error("WATCH_DIR is empty.");
        }

        FileProjectionSource source(config_.io);
        cv::Mat flat;
        cv::Mat dark;

        if (config_.io.correctionMode == CorrectionMode::Flat || config_.io.correctionMode == CorrectionMode::FlatDark)
        {
            emit logMessage(QStringLiteral("Waiting for flat reference..."));
            if (!source.waitForReferenceImage(config_.io.flatPrefix, QStringLiteral("Flat"), flat, stopRequested_))
            {
                if (stopRequested_.load()) { finishStopped(); return; }
                throw std::runtime_error("Failed to acquire flat reference.");
            }
        }

        if (config_.io.correctionMode == CorrectionMode::FlatDark)
        {
            emit logMessage(QStringLiteral("Waiting for dark reference..."));
            if (!source.waitForReferenceImage(config_.io.darkPrefix, QStringLiteral("Dark"), dark, stopRequested_))
            {
                if (stopRequested_.load()) { finishStopped(); return; }
                throw std::runtime_error("Failed to acquire dark reference.");
            }
        }

        cv::Mat ref0;
        cv::Mat ref180;
        QString ref0Path;
        QString ref180Path;

        emit logMessage(QStringLiteral("Waiting for 0 deg and 180 deg reference files..."));
        if (!source.waitForSpecificFile(config_.io.filePattern0, QStringLiteral("0 Deg"), ref0, &ref0Path, stopRequested_))
        {
            if (stopRequested_.load()) { finishStopped(); return; }
            throw std::runtime_error("Failed to acquire 0 degree reference.");
        }
        if (!source.waitForSpecificFile(config_.io.filePattern180, QStringLiteral("180 Deg"), ref180, &ref180Path, stopRequested_))
        {
            if (stopRequested_.load()) { finishStopped(); return; }
            throw std::runtime_error("Failed to acquire 180 degree reference.");
        }

        config_.geometry.volH = !flat.empty() ? flat.rows : ref0.rows;
        config_.geometry.volW = !flat.empty() ? flat.cols : ref0.cols;
        if (config_.geometry.corAuto)
        {
            emit logMessage(QStringLiteral("Estimating COR..."));
            CorEstimatorFft corEstimator;
            CorrectionContext correctionContext;
            correctionContext.mode = config_.io.correctionMode;
            correctionContext.flat = flat;
            correctionContext.dark = dark;
            correctionContext.cropMargin = 50;
            config_.geometry.corShift = corEstimator.estimateShift(ref0, ref180, correctionContext);
            emit corComputed(config_.geometry.corShift);
            emit logMessage(QStringLiteral("Auto COR Result: %1").arg(config_.geometry.corShift, 0, 'f', 4));
        }

        const auto allVectors = GeometryVectorGenerator::createParallel3dVectors(config_.geometry);

        QString cacheError;
        emit logMessage(QStringLiteral("Initializing CUDA runtime cache..."));
        if (!runtimeCache_->ensureInitialized(config_, flat, dark, &cacheError))
        {
            throw std::runtime_error(cacheError.toStdString());
        }
        emit logMessage(QStringLiteral("CUDA runtime cache initialized."));

        const size_t cpuVolumeBytes = volumeBytes(config_.geometry.volH, config_.geometry.volW, config_.geometry.volW);
        const size_t maxSlabHeight = static_cast<size_t>(std::min(config_.compute.zChunkSize, config_.geometry.volH));
        const size_t gpuSlabBytes = volumeBytes(static_cast<int>(maxSlabHeight), config_.geometry.volW, config_.geometry.volW);
        const size_t batchBytes = volumeBytes(config_.compute.batchSize, config_.geometry.volH, config_.geometry.volW);
        const size_t availableBytes = availablePhysicalMemoryBytes();
        emit logMessage(QStringLiteral("CPU full volume: %1 GiB float32. GPU slab: %2 GiB. Projection batch: %3 GiB.")
            .arg(formatGiB(cpuVolumeBytes), formatGiB(gpuSlabBytes), formatGiB(batchBytes)));
        emit logMessage(QStringLiteral("Available physical memory: %1 GiB.").arg(formatGiB(availableBytes)));

        constexpr size_t kCpuMemorySafetyMarginBytes = static_cast<size_t>(8) * 1024 * 1024 * 1024;
        if (availableBytes != std::numeric_limits<size_t>::max()
            && (cpuVolumeBytes == std::numeric_limits<size_t>::max()
                || cpuVolumeBytes + kCpuMemorySafetyMarginBytes > availableBytes))
        {
            throw std::runtime_error("Insufficient physical memory for CPU float32 volume.");
        }

        auto cpuVolume = std::make_shared<CpuVolumeBuffer>();
        emit logMessage(QStringLiteral("Allocating CPU float32 volume..."));
        QElapsedTimer allocationTimer;
        allocationTimer.start();
        if (!cpuVolume->initialize(config_.geometry.volH, config_.geometry.volW, config_.geometry.volW))
        {
            throw std::runtime_error("Failed to allocate CPU float32 volume.");
        }
        emit logMessage(QStringLiteral("CPU volume allocated in %1 s.").arg(allocationTimer.elapsed() / 1000.0, 0, 'f', 2));
        volumeBuffer_ = cpuVolume;

        emit logMessage(QStringLiteral("Allocating primary GPU slab..."));
        allocationTimer.restart();
        const size_t maxSlabElements = maxSlabHeight * static_cast<size_t>(config_.geometry.volW) * config_.geometry.volW;
        const bool usePinnedSlabPipeline = (maxSlabElements * sizeof(float)) <= kMaxPinnedSlabBytes;
        GpuVolumeBuffer slabVolume;
        if (!slabVolume.initialize(std::min(config_.compute.zChunkSize, config_.geometry.volH),
                config_.geometry.volW, config_.geometry.volW))
        {
            throw std::runtime_error("Failed to allocate GPU slab volume.");
        }
        emit logMessage(QStringLiteral("Primary GPU slab allocated in %1 s.").arg(allocationTimer.elapsed() / 1000.0, 0, 'f', 2));

        GpuVolumeBuffer slabVolumeAlt;
        bool doubleGpuSlabEnabled = false;
        if (usePinnedSlabPipeline)
        {
            emit logMessage(QStringLiteral("Trying to allocate secondary GPU slab for double buffering..."));
            allocationTimer.restart();
            doubleGpuSlabEnabled = slabVolumeAlt.initialize(std::min(config_.compute.zChunkSize, config_.geometry.volH),
                config_.geometry.volW, config_.geometry.volW);
            if (doubleGpuSlabEnabled)
            {
                emit logMessage(QStringLiteral("GPU slab pipeline: double-buffered GPU slabs enabled in %1 s.")
                    .arg(allocationTimer.elapsed() / 1000.0, 0, 'f', 2));
            }
            else
            {
                emit logMessage(QStringLiteral("GPU slab pipeline: second GPU slab allocation failed; using single GPU slab fallback."));
            }
        }
        else
        {
            emit logMessage(QStringLiteral("GPU slab pipeline: secondary GPU slab disabled for large direct-transfer slabs."));
        }

        PinnedFloatBuffer pinnedSlabA;
        PinnedFloatBuffer pinnedSlabB;
        if (usePinnedSlabPipeline)
        {
            emit logMessage(QStringLiteral("Allocating pinned host slab staging buffers..."));
            allocationTimer.restart();
            if (!pinnedSlabA.allocate(maxSlabElements) || !pinnedSlabB.allocate(maxSlabElements))
            {
                throw std::runtime_error("Failed to allocate pinned host slab buffers.");
            }
            emit logMessage(QStringLiteral("Pinned host slab buffers allocated in %1 s.").arg(allocationTimer.elapsed() / 1000.0, 0, 'f', 2));
        }
        else
        {
            emit logMessage(QStringLiteral("Pinned host slab buffers disabled for %1 GiB slab; using direct CPU volume transfers.")
                .arg(formatGiB(maxSlabElements * sizeof(float))));
        }

        SlabPipelineResources slabPipeline;
        if (usePinnedSlabPipeline && !slabPipeline.initialize())
        {
            throw std::runtime_error("Failed to create slab pipeline streams/events.");
        }

        previewSliceIndex_.store(std::clamp(previewSliceIndex_.load(), 0, config_.geometry.volH - 1));
        config_.preview.liveSliceIndex = previewSliceIndex_.load();
        emit volumeGeometryResolved(config_.geometry.volW, config_.geometry.volH);

        Bp3dCudaBackend backend;
        if (!backend.initialize(config_.geometry, config_.compute))
        {
            throw std::runtime_error("Failed to initialize CUDA backend.");
        }

        const int startIndex = source.extractStartIndex(config_.io.filePattern0);
        const int totalSlabs = (config_.geometry.volH + config_.compute.zChunkSize - 1) / config_.compute.zChunkSize;
        const int progressTotal = std::max(1, config_.geometry.totalAngles);
        std::vector<bool> processedFlags(static_cast<size_t>(config_.geometry.totalAngles), false);
        int processedAngles = 0;
        int batchCount = 0;
        bool firstLivePreviewPending = previewEnabledDuringRun;
        QElapsedTimer previewTimer;
        previewTimer.start();
        qint64 lastPreviewMs = -kLivePreviewMinIntervalMs;

        QElapsedTimer timer;
        timer.start();

        emit logMessage(QStringLiteral("Reconstruction mode: real-time batch-first CPU volume + GPU slab."));
        emit logMessage(QStringLiteral("Projection batch outer loop, %1 z slabs per batch.").arg(totalSlabs));

        while (processedAngles < config_.geometry.totalAngles && !stopRequested_.load())
        {
            auto batch = source.pollNextBatch(processedFlags, startIndex, config_.compute.batchSize, stopRequested_);
            if (stopRequested_.load())
            {
                break;
            }

            if (batch.empty())
            {
                QThread::msleep(10);
                continue;
            }

            auto& resources = runtimeCache_->slot(batchCount % runtimeCache_->slotCount());
            int validFrameCount = 0;
            std::vector<int> logicalIndices;
            logicalIndices.reserve(batch.size());
            resources.hostBatch.clear();
            QElapsedTimer frameStageTimer;
            frameStageTimer.start();

            for (const ProjectionFrame& frame : batch)
            {
                if (frame.image.empty())
                {
                    continue;
                }

                if (frame.logicalIndex < 0 || frame.logicalIndex >= config_.geometry.totalAngles)
                {
                    throw std::runtime_error("Projection logical index is outside configured angle range.");
                }

                if (frame.image.rows != config_.geometry.volH || frame.image.cols != config_.geometry.volW || frame.image.channels() != 1)
                {
                    throw std::runtime_error("Projection image size does not match configured volume geometry.");
                }

                if (!resources.hostBatch.copyFrame(validFrameCount, frame.image))
                {
                    throw std::runtime_error("Failed to stage projection frame into pinned host batch.");
                }

                logicalIndices.push_back(frame.logicalIndex);
                ++validFrameCount;
            }

            if (validFrameCount <= 0)
            {
                continue;
            }

            emit logMessage(QStringLiteral("Reconstructing batch %1: %2 projections, updating %3 z slabs.")
                .arg(batchCount + 1).arg(validFrameCount).arg(totalSlabs));
            if (!logicalIndices.empty())
            {
                emit logMessage(QStringLiteral("Batch %1 file range: logical %2..%3 staged in %4 ms.")
                    .arg(batchCount + 1)
                    .arg(logicalIndices.front())
                    .arg(logicalIndices.back())
                    .arg(frameStageTimer.elapsed()));
            }

            const cudaStream_t preprocessStream = runtimeCache_->preprocessStream(batchCount % runtimeCache_->slotCount());
            const auto batchVectors = sliceVectors(allVectors, logicalIndices);
            emit logMessage(QStringLiteral("Batch %1 GPU preprocess: upload raw projections.").arg(batchCount + 1));
            if (!resources.gpuBatch.uploadRawFromHostAsync(resources.hostBatch, validFrameCount, preprocessStream))
            {
                throw std::runtime_error("Failed to upload projection batch to GPU.");
            }
            if (cudaStreamSynchronize(preprocessStream) != cudaSuccess)
            {
                const cudaError_t status = cudaGetLastError();
                throw std::runtime_error(std::string("Failed to synchronize projection upload: ")
                    + cudaGetErrorString(status));
            }
            if (!resources.gpuVectors.uploadAsync(batchVectors, preprocessStream))
            {
                throw std::runtime_error("Failed to upload geometry vectors to GPU.");
            }
            emit logMessage(QStringLiteral("Batch %1 GPU preprocess: flat/dark correction.").arg(batchCount + 1));
            if (!ImageNormalizer::applyCorrection(resources.gpuBatch, runtimeCache_->correctionImages(),
                config_.io.correctionMode, preprocessStream))
            {
                const cudaError_t status = cudaGetLastError();
                throw std::runtime_error(std::string("GPU flat/dark correction failed: ")
                    + cudaGetErrorString(status));
            }
            if (cudaStreamSynchronize(preprocessStream) != cudaSuccess)
            {
                const cudaError_t status = cudaGetLastError();
                throw std::runtime_error(std::string("GPU flat/dark correction sync failed: ")
                    + cudaGetErrorString(status));
            }
            if (resources.paganin.isEnabled()
                && !resources.paganin.apply(resources.gpuBatch, preprocessStream))
            {
                throw std::runtime_error("GPU Paganin phase retrieval failed.");
            }
            if (resources.paganin.isEnabled() && cudaStreamSynchronize(preprocessStream) != cudaSuccess)
            {
                const cudaError_t status = cudaGetLastError();
                throw std::runtime_error(std::string("GPU Paganin phase retrieval sync failed: ")
                    + cudaGetErrorString(status));
            }
            emit logMessage(QStringLiteral("Batch %1 GPU preprocess: transpose.").arg(batchCount + 1));
            if (!ImageNormalizer::transposeRawToRecon(resources.gpuBatch, preprocessStream))
            {
                throw std::runtime_error("Failed to transpose projections into reconstruction layout.");
            }
            if (cudaStreamSynchronize(preprocessStream) != cudaSuccess)
            {
                const cudaError_t status = cudaGetLastError();
                throw std::runtime_error(std::string("GPU transpose sync failed: ")
                    + cudaGetErrorString(status));
            }
            emit logMessage(QStringLiteral("Batch %1 GPU preprocess: method/taper.").arg(batchCount + 1));
            if (!ImageNormalizer::applyMethodAndTaper(resources.gpuBatch, config_.preprocess, preprocessStream))
            {
                throw std::runtime_error("GPU preprocess method/taper failed.");
            }
            if (cudaStreamSynchronize(preprocessStream) != cudaSuccess)
            {
                const cudaError_t status = cudaGetLastError();
                throw std::runtime_error(std::string("GPU preprocess method/taper sync failed: ")
                    + cudaGetErrorString(status));
            }
            emit logMessage(QStringLiteral("Batch %1 GPU preprocess: FBP filter.").arg(batchCount + 1));
            if (!resources.filter.apply(resources.gpuBatch, preprocessStream))
            {
                throw std::runtime_error("GPU FBP filter failed.");
            }
            if (cudaStreamSynchronize(preprocessStream) != cudaSuccess)
            {
                const cudaError_t status = cudaGetLastError();
                throw std::runtime_error(std::string("GPU FBP filter sync failed: ")
                    + cudaGetErrorString(status));
            }
            emit logMessage(QStringLiteral("Batch %1 GPU preprocess complete.").arg(batchCount + 1));

            QElapsedTimer batchTimer;
            batchTimer.start();
            qint64 slabHostReadMs = 0;
            qint64 slabUploadIssueMs = 0;
            qint64 slabBackProjectIssueMs = 0;
            qint64 slabDownloadIssueMs = 0;
            qint64 slabWaitWriteMs = 0;

            const int slabCount = totalSlabs;
            std::vector<int> slabZStarts(static_cast<size_t>(slabCount), 0);
            std::vector<int> slabHeights(static_cast<size_t>(slabCount), 0);
            for (int slabIndex = 0; slabIndex < slabCount; ++slabIndex)
            {
                const int zStart = slabIndex * config_.compute.zChunkSize;
                slabZStarts[static_cast<size_t>(slabIndex)] = zStart;
                slabHeights[static_cast<size_t>(slabIndex)] = std::min(config_.compute.zChunkSize, config_.geometry.volH - zStart);
            }

            GpuVolumeBuffer* slabVolumes[2] = { &slabVolume, doubleGpuSlabEnabled ? &slabVolumeAlt : &slabVolume };
            PinnedFloatBuffer* pinnedSlabs[2] = { &pinnedSlabA, &pinnedSlabB };

            const auto processSlabDirect = [&](int slabIndex) {
                const int zStart = slabZStarts[static_cast<size_t>(slabIndex)];
                const int slabHeight = slabHeights[static_cast<size_t>(slabIndex)];
                float* cpuSlab = cpuVolume->rangePointer(zStart, slabHeight);
                if (cpuSlab == nullptr)
                {
                    throw std::runtime_error("Failed to get CPU slab pointer.");
                }

                if (slabIndex == 0 || ((slabIndex + 1) % 10) == 0 || slabIndex == slabCount - 1)
                {
                    emit logMessage(QStringLiteral("Batch %1 direct slab %2/%3: z=%4 h=%5.")
                        .arg(batchCount + 1)
                        .arg(slabIndex + 1)
                        .arg(slabCount)
                        .arg(zStart)
                        .arg(slabHeight));
                }

                QElapsedTimer stageTimer;
                stageTimer.start();
                if (processedAngles == 0)
                {
                    slabVolume.clear();
                }
                else if (!slabVolume.uploadAllFromHost(cpuSlab))
                {
                    throw std::runtime_error("Failed to upload CPU slab directly to GPU.");
                }
                slabHostReadMs += stageTimer.elapsed();

                stageTimer.restart();
                const DeviceChunkView chunk = resources.gpuBatch.makeChunkView(zStart, slabHeight);
                if (!backend.backProjectSlab(chunk, resources.gpuVectors, slabVolume, runtimeCache_->reconStream()))
                {
                    throw std::runtime_error("CUDA slab backprojection failed.");
                }
                if (cudaStreamSynchronize(runtimeCache_->reconStream()) != cudaSuccess)
                {
                    const cudaError_t status = cudaGetLastError();
                    throw std::runtime_error(std::string("CUDA slab backprojection sync failed: ")
                        + cudaGetErrorString(status));
                }
                slabBackProjectIssueMs += stageTimer.elapsed();

                stageTimer.restart();
                if (!slabVolume.downloadAllToHost(cpuSlab))
                {
                    throw std::runtime_error("Failed to download GPU slab directly to CPU volume.");
                }
                slabWaitWriteMs += stageTimer.elapsed();
            };

            const auto issueUpload = [&](int slabIndex) {
                const int slot = slabIndex % 2;
                const int zStart = slabZStarts[static_cast<size_t>(slabIndex)];
                const int slabHeight = slabHeights[static_cast<size_t>(slabIndex)];

                QElapsedTimer stageTimer;
                stageTimer.start();
                if (!cpuVolume->copyRangeTo(zStart, slabHeight, pinnedSlabs[slot]->data()))
                {
                    throw std::runtime_error("Failed to copy CPU slab into pinned staging buffer.");
                }
                slabHostReadMs += stageTimer.elapsed();

                stageTimer.restart();
                if (!slabVolumes[slot]->uploadRangeAsync(0, slabHeight, pinnedSlabs[slot]->data(),
                        slabPipeline.uploadStream()))
                {
                    throw std::runtime_error("Failed to enqueue CPU slab upload to GPU.");
                }
                if (cudaEventRecord(slabPipeline.uploadDone(slot), slabPipeline.uploadStream()) != cudaSuccess)
                {
                    throw std::runtime_error("Failed to record slab upload event.");
                }
                slabUploadIssueMs += stageTimer.elapsed();
            };

            const auto issueBackProjectAndDownload = [&](int slabIndex) {
                const int slot = slabIndex % 2;
                const int zStart = slabZStarts[static_cast<size_t>(slabIndex)];
                const int slabHeight = slabHeights[static_cast<size_t>(slabIndex)];

                QElapsedTimer stageTimer;
                stageTimer.start();
                if (cudaStreamWaitEvent(runtimeCache_->reconStream(), slabPipeline.uploadDone(slot), 0) != cudaSuccess)
                {
                    throw std::runtime_error("Failed to wait for slab upload before backprojection.");
                }
                const DeviceChunkView chunk = resources.gpuBatch.makeChunkView(zStart, slabHeight);
                if (!backend.backProjectSlab(chunk, resources.gpuVectors, *slabVolumes[slot], runtimeCache_->reconStream()))
                {
                    throw std::runtime_error("CUDA slab backprojection failed.");
                }
                if (cudaEventRecord(slabPipeline.reconDone(slot), runtimeCache_->reconStream()) != cudaSuccess)
                {
                    throw std::runtime_error("Failed to record slab reconstruction event.");
                }
                slabBackProjectIssueMs += stageTimer.elapsed();

                stageTimer.restart();
                if (cudaStreamWaitEvent(slabPipeline.downloadStream(), slabPipeline.reconDone(slot), 0) != cudaSuccess)
                {
                    throw std::runtime_error("Failed to wait for slab reconstruction before download.");
                }
                if (!slabVolumes[slot]->downloadRangeAsync(0, slabHeight, pinnedSlabs[slot]->data(),
                        slabPipeline.downloadStream()))
                {
                    throw std::runtime_error("Failed to enqueue updated GPU slab download.");
                }
                if (cudaEventRecord(slabPipeline.downloadDone(slot), slabPipeline.downloadStream()) != cudaSuccess)
                {
                    throw std::runtime_error("Failed to record slab download event.");
                }
                slabDownloadIssueMs += stageTimer.elapsed();
            };

            const auto waitAndWriteBack = [&](int slabIndex) {
                const int slot = slabIndex % 2;
                const int zStart = slabZStarts[static_cast<size_t>(slabIndex)];
                const int slabHeight = slabHeights[static_cast<size_t>(slabIndex)];

                QElapsedTimer stageTimer;
                stageTimer.start();
                if (cudaEventSynchronize(slabPipeline.downloadDone(slot)) != cudaSuccess)
                {
                    throw std::runtime_error("Failed to synchronize updated slab download.");
                }
                if (!cpuVolume->writeSlabFrom(zStart, slabHeight, pinnedSlabs[slot]->data()))
                {
                    throw std::runtime_error("Failed to write updated pinned slab into CPU volume.");
                }
                slabWaitWriteMs += stageTimer.elapsed();
            };

            if (!usePinnedSlabPipeline)
            {
                for (int slabIndex = 0; slabIndex < slabCount && !stopRequested_.load(); ++slabIndex)
                {
                    processSlabDirect(slabIndex);
                }
            }
            else if (slabCount > 0 && doubleGpuSlabEnabled)
            {
                issueUpload(0);

                for (int slabIndex = 0; slabIndex < slabCount && !stopRequested_.load(); ++slabIndex)
                {
                    issueBackProjectAndDownload(slabIndex);

                    const int nextSlab = slabIndex + 1;
                    if (nextSlab < slabCount)
                    {
                        if (nextSlab >= 2)
                        {
                            waitAndWriteBack(nextSlab - 2);
                        }
                        issueUpload(nextSlab);
                    }
                    else if (slabIndex >= 1)
                    {
                        waitAndWriteBack(slabIndex - 1);
                    }
                }

                if (!stopRequested_.load())
                {
                    waitAndWriteBack(slabCount - 1);
                }
            }
            else if (slabCount > 0)
            {
                for (int slabIndex = 0; slabIndex < slabCount && !stopRequested_.load(); ++slabIndex)
                {
                    issueUpload(slabIndex);
                    issueBackProjectAndDownload(slabIndex);
                    waitAndWriteBack(slabIndex);
                }
            }

            if (stopRequested_.load())
            {
                break;
            }

            for (const int logicalIndex : logicalIndices)
            {
                processedFlags[static_cast<size_t>(logicalIndex)] = true;
            }
            processedAngles += validFrameCount;
            ++batchCount;
            emit progressUpdated(std::min(processedAngles, config_.geometry.totalAngles), progressTotal);
            emit logMessage(QStringLiteral("Batch %1 timings: total=%2 ms, slabReadOrUpload=%3 ms, h2dIssue=%4 ms, backProject=%5 ms, d2hIssue=%6 ms, slabDownloadOrWrite=%7 ms.")
                .arg(batchCount)
                .arg(batchTimer.elapsed())
                .arg(slabHostReadMs)
                .arg(slabUploadIssueMs)
                .arg(slabBackProjectIssueMs)
                .arg(slabDownloadIssueMs)
                .arg(slabWaitWriteMs));

            if (previewEnabledDuringRun)
            {
                const qint64 nowMs = previewTimer.elapsed();
                const bool shouldInitializePreview = firstLivePreviewPending;
                const bool shouldEmitPeriodicPreview =
                    (batchCount % kLivePreviewEveryNBatches) == 0
                    && (nowMs - lastPreviewMs) >= kLivePreviewMinIntervalMs;

                if (shouldInitializePreview || shouldEmitPeriodicPreview)
                {
                    emit previewRefreshSuggested(shouldInitializePreview);
                    firstLivePreviewPending = false;
                    lastPreviewMs = nowMs;
                }
            }
        }

        if (stopRequested_.load())
        {
            finishStopped();
            return;
        }

        if (processedAngles < config_.geometry.totalAngles)
        {
            throw std::runtime_error("Not all projection angles were available for reconstruction.");
        }

        if (cudaStreamSynchronize(runtimeCache_->reconStream()) != cudaSuccess)
        {
            throw std::runtime_error("Failed to synchronize reconstruction stream.");
        }

        emit logMessage(QStringLiteral("Reconstruction finished in %1 s").arg(timer.elapsed() / 1000.0, 0, 'f', 2));
        emit reconstructionFinished();
    }
    catch (const std::exception& ex)
    {
        appendReconDebug(QStringLiteral("worker caught exception: %1").arg(QString::fromLocal8Bit(ex.what())));
        emit reconstructionFailed(QString::fromLocal8Bit(ex.what()));
    }
}

} // namespace ct
