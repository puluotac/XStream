#pragma once

#include "core/config/AppConfig.h"
#include "core/preprocess/CudaFftPlanCache.h"
#include "core/recon/GpuProjectionBatch.h"

namespace ct
{

class FbpFilter1D
{
public:
    FbpFilter1D() = default;
    ~FbpFilter1D();

    bool configure(int detectorWidth, int angleCapacity, FilterType type);
    bool apply(GpuProjectionBatch& batch, cudaStream_t stream);

private:
    bool ensureWorkspace(int transformCount);

    int detectorWidth_ = 0;
    int paddedLength_ = 0;
    int angleCapacity_ = 0;
    FilterType filterType_ = FilterType::SheppLogan;
    CudaFftPlanCache planCache_;
    float* deviceFilterSpectrum_ = nullptr;
    float* devicePaddedReal_ = nullptr;
    cufftComplex* deviceSpectrum_ = nullptr;
    size_t paddedCapacity_ = 0;
    size_t spectrumCapacity_ = 0;
};

} // namespace ct
