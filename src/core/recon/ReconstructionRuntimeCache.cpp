#include "core/recon/ReconstructionRuntimeCache.h"

#include <cuda_runtime.h>

namespace ct
{

namespace
{

bool phaseConfigEquals(const PhaseConfig& left, const PhaseConfig& right)
{
    return left.enable == right.enable
        && left.energyKev == right.energyKev
        && left.distanceM == right.distanceM
        && left.pixelSizeUm == right.pixelSizeUm
        && left.alphaRatio == right.alphaRatio;
}

} // namespace

bool ReconstructionRuntimeCache::SlotResources::ensureInitialized(
    int batchSize,
    int height,
    int width,
    QString* errorMessage)
{
    if (!hostBatch.initialize(batchSize, height, width))
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to allocate pinned host batch.");
        }
        return false;
    }

    if (!gpuBatch.initialize(batchSize, height, width))
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to allocate GPU projection batch.");
        }
        return false;
    }

    if (!gpuVectors.initialize(batchSize))
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to allocate GPU vector buffer.");
        }
        return false;
    }

    if (preprocessDone == nullptr && cudaEventCreateWithFlags(&preprocessDone, cudaEventDisableTiming) != cudaSuccess)
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to create preprocess CUDA event.");
        }
        return false;
    }

    if (reconDone == nullptr && cudaEventCreateWithFlags(&reconDone, cudaEventDisableTiming) != cudaSuccess)
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to create reconstruction CUDA event.");
        }
        return false;
    }

    if (preprocessStream == nullptr
        && cudaStreamCreateWithFlags(&preprocessStream, cudaStreamNonBlocking) != cudaSuccess)
    {
        preprocessStream = nullptr;
    }

    return true;
}

void ReconstructionRuntimeCache::SlotResources::release()
{
    if (preprocessStream != nullptr)
    {
        cudaStreamDestroy(preprocessStream);
        preprocessStream = nullptr;
    }
    if (preprocessDone != nullptr)
    {
        cudaEventDestroy(preprocessDone);
        preprocessDone = nullptr;
    }
    if (reconDone != nullptr)
    {
        cudaEventDestroy(reconDone);
        reconDone = nullptr;
    }
}

ReconstructionRuntimeCache::~ReconstructionRuntimeCache()
{
    release();
}

bool ReconstructionRuntimeCache::ensureInitialized(
    const AppConfig& config,
    const cv::Mat& flat,
    const cv::Mat& dark,
    QString* errorMessage)
{
    if (cudaSetDevice(config.compute.gpuIndex) != cudaSuccess)
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to select CUDA device for runtime cache.");
        }
        return false;
    }

    if (!ensureStreams(errorMessage))
    {
        return false;
    }

    if (!ensureProcessors(config, errorMessage))
    {
        return false;
    }

    if (!isLayoutCompatible(config))
    {
        for (SlotResources& slot : slots_)
        {
            if (!slot.ensureInitialized(config.compute.batchSize, config.geometry.volH, config.geometry.volW, errorMessage))
            {
                return false;
            }
        }

        cachedVolW_ = config.geometry.volW;
        cachedVolH_ = config.geometry.volH;
        cachedBatchSize_ = config.compute.batchSize;
        cachedGpuIndex_ = config.compute.gpuIndex;
    }

    if (!ImageNormalizer::uploadCorrectionImages(flat, dark,
        config.geometry.volH, config.geometry.volW, correctionImages_))
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to upload flat/dark correction images.");
        }
        return false;
    }

    return true;
}

void ReconstructionRuntimeCache::release()
{
    for (SlotResources& slot : slots_)
    {
        slot.release();
    }

    correctionImages_.release();

    if (fallbackPreprocessStream_ != nullptr)
    {
        cudaStreamDestroy(fallbackPreprocessStream_);
        fallbackPreprocessStream_ = nullptr;
    }
    if (reconStream_ != nullptr)
    {
        cudaStreamDestroy(reconStream_);
        reconStream_ = nullptr;
    }

    cachedVolW_ = 0;
    cachedVolH_ = 0;
    cachedBatchSize_ = 0;
    cachedGpuIndex_ = -1;
    hasPhaseConfig_ = false;
    hasFilterConfig_ = false;
}

int ReconstructionRuntimeCache::slotCount() const
{
    return static_cast<int>(slots_.size());
}

ReconstructionRuntimeCache::SlotResources& ReconstructionRuntimeCache::slot(int index)
{
    return slots_[static_cast<size_t>(index)];
}

const ReconstructionRuntimeCache::SlotResources& ReconstructionRuntimeCache::slot(int index) const
{
    return slots_[static_cast<size_t>(index)];
}

const DeviceCorrectionImages& ReconstructionRuntimeCache::correctionImages() const
{
    return correctionImages_;
}

cudaStream_t ReconstructionRuntimeCache::preprocessStream(int slotIndex) const
{
    if (slotIndex >= 0 && slotIndex < static_cast<int>(slots_.size()))
    {
        const cudaStream_t slotStream = slots_[static_cast<size_t>(slotIndex)].preprocessStream;
        if (slotStream != nullptr)
        {
            return slotStream;
        }
    }

    return fallbackPreprocessStream_;
}

cudaStream_t ReconstructionRuntimeCache::reconStream() const
{
    return reconStream_;
}

bool ReconstructionRuntimeCache::ensureStreams(QString* errorMessage)
{
    if (fallbackPreprocessStream_ == nullptr
        && cudaStreamCreateWithFlags(&fallbackPreprocessStream_, cudaStreamNonBlocking) != cudaSuccess)
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to create preprocess CUDA stream.");
        }
        return false;
    }

    if (reconStream_ == nullptr
        && cudaStreamCreateWithFlags(&reconStream_, cudaStreamNonBlocking) != cudaSuccess)
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to create reconstruction CUDA stream.");
        }
        return false;
    }

    return true;
}

bool ReconstructionRuntimeCache::ensureProcessors(const AppConfig& config, QString* errorMessage)
{
    if (!hasPhaseConfig_ || !phaseConfigEquals(cachedPhaseConfig_, config.phase))
    {
        for (SlotResources& slot : slots_)
        {
            slot.paganin.configure(config.phase);
        }
        cachedPhaseConfig_ = config.phase;
        hasPhaseConfig_ = true;
    }

    if (!hasFilterConfig_
        || cachedFilterType_ != config.preprocess.filterType
        || cachedVolW_ != config.geometry.volW
        || cachedBatchSize_ != config.compute.batchSize)
    {
        for (SlotResources& slot : slots_)
        {
            if (!slot.filter.configure(config.geometry.volW, config.compute.batchSize, config.preprocess.filterType))
            {
                if (errorMessage != nullptr)
                {
                    *errorMessage = QStringLiteral("Failed to configure cached GPU FBP filter.");
                }
                return false;
            }
        }

        cachedFilterType_ = config.preprocess.filterType;
        hasFilterConfig_ = true;
    }

    return true;
}

bool ReconstructionRuntimeCache::isLayoutCompatible(const AppConfig& config) const
{
    return cachedVolW_ == config.geometry.volW
        && cachedVolH_ == config.geometry.volH
        && cachedBatchSize_ == config.compute.batchSize
        && cachedGpuIndex_ == config.compute.gpuIndex;
}

} // namespace ct
