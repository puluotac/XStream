#pragma once

#include "core/config/AppConfig.h"
#include "core/preprocess/FbpFilter1D.h"
#include "core/preprocess/ImageNormalizer.h"
#include "core/preprocess/PaganinProcessor.h"
#include "core/recon/GpuProjectionBatch.h"
#include "core/recon/GpuVectorBuffer.h"

#include <QString>

#include <opencv2/core.hpp>

#include <array>

#include <cuda_runtime_api.h>

namespace ct
{

class ReconstructionRuntimeCache
{
public:
    struct SlotResources
    {
        HostPinnedProjectionBatch hostBatch;
        GpuProjectionBatch gpuBatch;
        GpuVectorBuffer gpuVectors;
        PaganinProcessor paganin;
        FbpFilter1D filter;
        cudaStream_t preprocessStream = nullptr;
        cudaEvent_t preprocessDone = nullptr;
        cudaEvent_t reconDone = nullptr;

        bool ensureInitialized(int batchSize, int height, int width, QString* errorMessage);
        void release();
    };

    ReconstructionRuntimeCache() = default;
    ~ReconstructionRuntimeCache();

    ReconstructionRuntimeCache(const ReconstructionRuntimeCache&) = delete;
    ReconstructionRuntimeCache& operator=(const ReconstructionRuntimeCache&) = delete;

    bool ensureInitialized(const AppConfig& config, const cv::Mat& flat, const cv::Mat& dark, QString* errorMessage);
    void release();

    int slotCount() const;
    SlotResources& slot(int index);
    const SlotResources& slot(int index) const;

    const DeviceCorrectionImages& correctionImages() const;

    cudaStream_t preprocessStream(int slotIndex) const;
    cudaStream_t reconStream() const;

private:
    bool ensureStreams(QString* errorMessage);
    bool ensureProcessors(const AppConfig& config, QString* errorMessage);
    bool isLayoutCompatible(const AppConfig& config) const;

    std::array<SlotResources, 1> slots_;
    DeviceCorrectionImages correctionImages_;
    cudaStream_t fallbackPreprocessStream_ = nullptr;
    cudaStream_t reconStream_ = nullptr;

    int cachedVolW_ = 0;
    int cachedVolH_ = 0;
    int cachedBatchSize_ = 0;
    int cachedGpuIndex_ = -1;
    PhaseConfig cachedPhaseConfig_;
    bool hasPhaseConfig_ = false;
    FilterType cachedFilterType_ = FilterType::SheppLogan;
    bool hasFilterConfig_ = false;
};

} // namespace ct
