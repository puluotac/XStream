#include "core/preprocess/CudaFftPlanCache.h"

namespace ct
{

CudaFftPlanCache::~CudaFftPlanCache()
{
    destroyPaganin();
    destroyFbp();
}

void CudaFftPlanCache::setStream(cudaStream_t stream)
{
    stream_ = stream;

    if (paganinForward_ != 0) { cufftSetStream(paganinForward_, stream_); }
    if (paganinInverse_ != 0) { cufftSetStream(paganinInverse_, stream_); }
    if (fbpForward_ != 0) { cufftSetStream(fbpForward_, stream_); }
    if (fbpInverse_ != 0) { cufftSetStream(fbpInverse_, stream_); }
}

bool CudaFftPlanCache::ensurePaganinPlans(int batchAngles, int paddedHeight, int width)
{
    if (paganinForward_ != 0 && paganinInverse_ != 0
        && paganinBatchAngles_ == batchAngles && paganinPaddedHeight_ == paddedHeight && paganinWidth_ == width)
    {
        return true;
    }

    destroyPaganin();

    const int rank = 2;
    int n[2] = { paddedHeight, width };
    int inembed[2] = { paddedHeight, width };
    int onembed[2] = { paddedHeight, (width / 2) + 1 };

    if (cufftPlanMany(&paganinForward_, rank, n,
        inembed, 1, paddedHeight * width,
        onembed, 1, paddedHeight * ((width / 2) + 1),
        CUFFT_R2C, batchAngles) != CUFFT_SUCCESS)
    {
        destroyPaganin();
        return false;
    }

    if (cufftPlanMany(&paganinInverse_, rank, n,
        onembed, 1, paddedHeight * ((width / 2) + 1),
        inembed, 1, paddedHeight * width,
        CUFFT_C2R, batchAngles) != CUFFT_SUCCESS)
    {
        destroyPaganin();
        return false;
    }

    setStream(stream_);
    paganinBatchAngles_ = batchAngles;
    paganinPaddedHeight_ = paddedHeight;
    paganinWidth_ = width;
    return true;
}

bool CudaFftPlanCache::ensureFbpPlans(int transformCount, int paddedLength)
{
    if (fbpForward_ != 0 && fbpInverse_ != 0
        && fbpTransformCount_ == transformCount && fbpPaddedLength_ == paddedLength)
    {
        return true;
    }

    destroyFbp();

    const int rank = 1;
    int n[1] = { paddedLength };
    int inembed[1] = { paddedLength };
    int onembed[1] = { (paddedLength / 2) + 1 };

    if (cufftPlanMany(&fbpForward_, rank, n,
        inembed, 1, paddedLength,
        onembed, 1, (paddedLength / 2) + 1,
        CUFFT_R2C, transformCount) != CUFFT_SUCCESS)
    {
        destroyFbp();
        return false;
    }

    if (cufftPlanMany(&fbpInverse_, rank, n,
        onembed, 1, (paddedLength / 2) + 1,
        inembed, 1, paddedLength,
        CUFFT_C2R, transformCount) != CUFFT_SUCCESS)
    {
        destroyFbp();
        return false;
    }

    setStream(stream_);
    fbpTransformCount_ = transformCount;
    fbpPaddedLength_ = paddedLength;
    return true;
}

cufftHandle CudaFftPlanCache::paganinForward() const
{
    return paganinForward_;
}

cufftHandle CudaFftPlanCache::paganinInverse() const
{
    return paganinInverse_;
}

cufftHandle CudaFftPlanCache::fbpForward() const
{
    return fbpForward_;
}

cufftHandle CudaFftPlanCache::fbpInverse() const
{
    return fbpInverse_;
}

void CudaFftPlanCache::destroyPaganin()
{
    if (paganinForward_ != 0)
    {
        cufftDestroy(paganinForward_);
        paganinForward_ = 0;
    }
    if (paganinInverse_ != 0)
    {
        cufftDestroy(paganinInverse_);
        paganinInverse_ = 0;
    }
    paganinBatchAngles_ = 0;
    paganinPaddedHeight_ = 0;
    paganinWidth_ = 0;
}

void CudaFftPlanCache::destroyFbp()
{
    if (fbpForward_ != 0)
    {
        cufftDestroy(fbpForward_);
        fbpForward_ = 0;
    }
    if (fbpInverse_ != 0)
    {
        cufftDestroy(fbpInverse_);
        fbpInverse_ = 0;
    }
    fbpTransformCount_ = 0;
    fbpPaddedLength_ = 0;
}

} // namespace ct
