#pragma once

#include "core/config/AppConfig.h"
#include "core/preprocess/CudaFftPlanCache.h"
#include "core/recon/GpuProjectionBatch.h"

namespace ct
{

class PaganinProcessor
{
public:
    PaganinProcessor() = default;
    ~PaganinProcessor();

    void configure(const PhaseConfig& config);
    bool isEnabled() const;
    bool apply(GpuProjectionBatch& batch, cudaStream_t stream);

private:
    void ensureKernel(int paddedHeight, int width);
    void ensurePaddingLayout(int height);
    bool ensureWorkspaceCapacity(int batchAngles, int paddedHeight, int width);

    PhaseConfig config_;
    CudaFftPlanCache planCache_;
    float* deviceFilterKernel_ = nullptr;
    float* devicePaddedReal_ = nullptr;
    cufftComplex* deviceSpectrum_ = nullptr;
    size_t paddedRealCapacity_ = 0;
    size_t spectrumCapacity_ = 0;
    int cachedHeight_ = 0;
    int cachedWidth_ = 0;
    int paddedHeight_ = 0;
    int padTop_ = 0;
    int padBottom_ = 0;
};

} // namespace ct
