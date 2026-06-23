#include "core/preprocess/FbpFilter1D.h"

#include "cuda/preprocess_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ct
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

} // namespace

FbpFilter1D::~FbpFilter1D()
{
    if (deviceFilterSpectrum_ != nullptr)
    {
        cudaFree(deviceFilterSpectrum_);
        deviceFilterSpectrum_ = nullptr;
    }
    if (devicePaddedReal_ != nullptr)
    {
        cudaFree(devicePaddedReal_);
        devicePaddedReal_ = nullptr;
    }
    if (deviceSpectrum_ != nullptr)
    {
        cudaFree(deviceSpectrum_);
        deviceSpectrum_ = nullptr;
    }
}

bool FbpFilter1D::configure(int detectorWidth, int angleCapacity, FilterType type)
{
    detectorWidth_ = detectorWidth;
    angleCapacity_ = angleCapacity;
    filterType_ = type;
    paddedLength_ = 1;
    while (paddedLength_ < detectorWidth_ * 2)
    {
        paddedLength_ <<= 1;
    }

    if (deviceFilterSpectrum_ != nullptr)
    {
        cudaFree(deviceFilterSpectrum_);
        deviceFilterSpectrum_ = nullptr;
    }

    const int spectrumLength = (paddedLength_ / 2) + 1;
    std::vector<float> hostSpectrum(static_cast<size_t>(spectrumLength), 0.0f);
    for (int i = 0; i < spectrumLength; ++i)
    {
        const double freq = static_cast<double>(i) / paddedLength_;
        const double w = freq * 2.0;
        double window = 1.0;

        switch (filterType_)
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

        hostSpectrum[static_cast<size_t>(i)] = static_cast<float>(std::abs(freq) * window);
    }

    if (cudaMalloc(&deviceFilterSpectrum_, hostSpectrum.size() * sizeof(float)) != cudaSuccess)
    {
        deviceFilterSpectrum_ = nullptr;
        return false;
    }

    if (cudaMemcpy(deviceFilterSpectrum_, hostSpectrum.data(), hostSpectrum.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(deviceFilterSpectrum_);
        deviceFilterSpectrum_ = nullptr;
        return false;
    }

    paddedCapacity_ = 0;
    spectrumCapacity_ = 0;
    return true;
}

bool FbpFilter1D::ensureWorkspace(int transformCount)
{
    const size_t paddedCount = static_cast<size_t>(transformCount) * paddedLength_;
    const size_t spectrumCount = static_cast<size_t>(transformCount) * ((paddedLength_ / 2) + 1);

    if (paddedCount > paddedCapacity_)
    {
        if (devicePaddedReal_ != nullptr)
        {
            cudaFree(devicePaddedReal_);
            devicePaddedReal_ = nullptr;
        }

        if (cudaMalloc(&devicePaddedReal_, paddedCount * sizeof(float)) != cudaSuccess)
        {
            devicePaddedReal_ = nullptr;
            paddedCapacity_ = 0;
            return false;
        }

        paddedCapacity_ = paddedCount;
    }

    if (spectrumCount > spectrumCapacity_)
    {
        if (deviceSpectrum_ != nullptr)
        {
            cudaFree(deviceSpectrum_);
            deviceSpectrum_ = nullptr;
        }

        if (cudaMalloc(&deviceSpectrum_, spectrumCount * sizeof(cufftComplex)) != cudaSuccess)
        {
            deviceSpectrum_ = nullptr;
            spectrumCapacity_ = 0;
            return false;
        }

        spectrumCapacity_ = spectrumCount;
    }

    return true;
}

bool FbpFilter1D::apply(GpuProjectionBatch& batch, cudaStream_t stream)
{
    if (batch.reconLayout() == nullptr || batch.currentAngleCount() <= 0 || detectorWidth_ <= 0 || paddedLength_ <= 0
        || deviceFilterSpectrum_ == nullptr)
    {
        return false;
    }

    const int transformCount = batch.height() * batch.currentAngleCount();
    if (!ensureWorkspace(transformCount))
    {
        return false;
    }

    planCache_.setStream(stream);
    if (!planCache_.ensureFbpPlans(transformCount, paddedLength_))
    {
        return false;
    }

    if (!launchPrepareFbpInput(batch.reconLayout(), devicePaddedReal_,
        transformCount, detectorWidth_, paddedLength_, stream))
    {
        return false;
    }

    if (cufftExecR2C(planCache_.fbpForward(),
        reinterpret_cast<cufftReal*>(devicePaddedReal_), deviceSpectrum_) != CUFFT_SUCCESS)
    {
        return false;
    }

    if (!launchMultiplyFbpSpectrum(deviceSpectrum_, deviceFilterSpectrum_,
        transformCount, (paddedLength_ / 2) + 1, stream))
    {
        return false;
    }

    if (cufftExecC2R(planCache_.fbpInverse(),
        deviceSpectrum_, reinterpret_cast<cufftReal*>(devicePaddedReal_)) != CUFFT_SUCCESS)
    {
        return false;
    }

    return launchExtractFbpOutput(devicePaddedReal_, batch.reconLayout(),
        transformCount, detectorWidth_, paddedLength_, static_cast<float>(paddedLength_), stream);
}

} // namespace ct
