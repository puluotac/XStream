#pragma once

#include <cuda_runtime_api.h>
#include <cufft.h>

namespace ct
{

class CudaFftPlanCache
{
public:
    CudaFftPlanCache() = default;
    ~CudaFftPlanCache();

    CudaFftPlanCache(const CudaFftPlanCache&) = delete;
    CudaFftPlanCache& operator=(const CudaFftPlanCache&) = delete;

    void setStream(cudaStream_t stream);

    bool ensurePaganinPlans(int batchAngles, int paddedHeight, int width);
    bool ensureFbpPlans(int transformCount, int paddedLength);

    cufftHandle paganinForward() const;
    cufftHandle paganinInverse() const;
    cufftHandle fbpForward() const;
    cufftHandle fbpInverse() const;

private:
    void destroyPaganin();
    void destroyFbp();

    cudaStream_t stream_ = nullptr;
    cufftHandle paganinForward_ = 0;
    cufftHandle paganinInverse_ = 0;
    cufftHandle fbpForward_ = 0;
    cufftHandle fbpInverse_ = 0;
    int paganinBatchAngles_ = 0;
    int paganinPaddedHeight_ = 0;
    int paganinWidth_ = 0;
    int fbpTransformCount_ = 0;
    int fbpPaddedLength_ = 0;
};

} // namespace ct
