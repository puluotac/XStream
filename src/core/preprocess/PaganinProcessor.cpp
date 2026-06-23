#include "core/preprocess/PaganinProcessor.h"

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
constexpr int kMinimumPaganinPadPerSide = 16;
constexpr int kPreferredPaganinHeightMultiple = 32;

int roundUpToMultiple(int value, int multiple)
{
    if (multiple <= 1)
    {
        return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

PaganinProcessor::~PaganinProcessor()
{
    if (deviceFilterKernel_ != nullptr)
    {
        cudaFree(deviceFilterKernel_);
        deviceFilterKernel_ = nullptr;
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

void PaganinProcessor::configure(const PhaseConfig& config)
{
    config_ = config;
    cachedHeight_ = 0;
    cachedWidth_ = 0;
    paddedHeight_ = 0;
    padTop_ = 0;
    padBottom_ = 0;
    paddedRealCapacity_ = 0;
    spectrumCapacity_ = 0;

    if (deviceFilterKernel_ != nullptr)
    {
        cudaFree(deviceFilterKernel_);
        deviceFilterKernel_ = nullptr;
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

bool PaganinProcessor::isEnabled() const
{
    return config_.enable;
}

void PaganinProcessor::ensurePaddingLayout(int height)
{
    const int targetPadPerSide = std::max(kMinimumPaganinPadPerSide, height / 20);
    int targetHeight = height + 2 * targetPadPerSide;
    targetHeight = roundUpToMultiple(targetHeight, kPreferredPaganinHeightMultiple);
    if (targetHeight <= height)
    {
        targetHeight = height + 2 * kMinimumPaganinPadPerSide;
    }

    paddedHeight_ = targetHeight;
    const int totalPad = paddedHeight_ - height;
    padTop_ = totalPad / 2;
    padBottom_ = totalPad - padTop_;
}

void PaganinProcessor::ensureKernel(int paddedHeight, int width)
{
    if (cachedHeight_ == paddedHeight && cachedWidth_ == width && deviceFilterKernel_ != nullptr)
    {
        return;
    }

    if (deviceFilterKernel_ != nullptr)
    {
        cudaFree(deviceFilterKernel_);
        deviceFilterKernel_ = nullptr;
    }

    const int spectrumWidth = (width / 2) + 1;
    std::vector<float> hostKernel(static_cast<size_t>(paddedHeight) * spectrumWidth, 1.0f);

    const double wavelength = (12.3984 / std::max(1.0f, config_.energyKev)) * 1e-10;
    const double factor = kPi * wavelength * config_.distanceM * config_.alphaRatio;
    const double pixelSizeM = std::max(1e-12f, config_.pixelSizeUm * 1e-6f);

    for (int y = 0; y < paddedHeight; ++y)
    {
        const double v = ((y <= paddedHeight / 2) ? y : y - paddedHeight) / (paddedHeight * pixelSizeM);
        for (int x = 0; x < spectrumWidth; ++x)
        {
            const double u = static_cast<double>(x) / (width * pixelSizeM);
            hostKernel[static_cast<size_t>(y) * spectrumWidth + x] =
                static_cast<float>(1.0 / (1.0 + factor * (u * u + v * v)));
        }
    }

    if (cudaMalloc(&deviceFilterKernel_, hostKernel.size() * sizeof(float)) != cudaSuccess)
    {
        deviceFilterKernel_ = nullptr;
        return;
    }

    if (cudaMemcpy(deviceFilterKernel_, hostKernel.data(), hostKernel.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(deviceFilterKernel_);
        deviceFilterKernel_ = nullptr;
        return;
    }

    cachedHeight_ = paddedHeight;
    cachedWidth_ = width;
}

bool PaganinProcessor::ensureWorkspaceCapacity(int batchAngles, int paddedHeight, int width)
{
    const size_t requiredRealCount = static_cast<size_t>(batchAngles) * paddedHeight * width;
    if (requiredRealCount > paddedRealCapacity_ || devicePaddedReal_ == nullptr)
    {
        if (devicePaddedReal_ != nullptr)
        {
            cudaFree(devicePaddedReal_);
            devicePaddedReal_ = nullptr;
        }

        if (cudaMalloc(&devicePaddedReal_, requiredRealCount * sizeof(float)) != cudaSuccess)
        {
            devicePaddedReal_ = nullptr;
            paddedRealCapacity_ = 0;
            return false;
        }

        paddedRealCapacity_ = requiredRealCount;
    }

    const size_t requiredSpectrumCount = static_cast<size_t>(batchAngles) * paddedHeight * ((width / 2) + 1);
    if (requiredSpectrumCount <= spectrumCapacity_ && deviceSpectrum_ != nullptr)
    {
        return true;
    }

    if (deviceSpectrum_ != nullptr)
    {
        cudaFree(deviceSpectrum_);
        deviceSpectrum_ = nullptr;
    }

    if (cudaMalloc(&deviceSpectrum_, requiredSpectrumCount * sizeof(cufftComplex)) != cudaSuccess)
    {
        deviceSpectrum_ = nullptr;
        spectrumCapacity_ = 0;
        return false;
    }

    spectrumCapacity_ = requiredSpectrumCount;
    return true;
}

bool PaganinProcessor::apply(GpuProjectionBatch& batch, cudaStream_t stream)
{
    if (!isEnabled())
    {
        return true;
    }

    if (batch.rawAngleMajor() == nullptr || batch.currentAngleCount() <= 0)
    {
        return false;
    }

    ensurePaddingLayout(batch.height());
    ensureKernel(paddedHeight_, batch.width());
    if (deviceFilterKernel_ == nullptr)
    {
        return false;
    }

    if (!ensureWorkspaceCapacity(batch.currentAngleCount(), paddedHeight_, batch.width()))
    {
        return false;
    }

    planCache_.setStream(stream);
    if (!planCache_.ensurePaganinPlans(batch.currentAngleCount(), paddedHeight_, batch.width()))
    {
        return false;
    }

    if (!launchPadPaganinInputReflectY(batch.rawAngleMajor(), devicePaddedReal_,
        batch.currentAngleCount(), batch.height(), paddedHeight_, batch.width(), padTop_, stream))
    {
        return false;
    }

    if (cufftExecR2C(planCache_.paganinForward(),
        reinterpret_cast<cufftReal*>(devicePaddedReal_), deviceSpectrum_) != CUFFT_SUCCESS)
    {
        return false;
    }

    if (!launchMultiplyPaganinSpectrum(deviceSpectrum_, deviceFilterKernel_, batch.currentAngleCount(),
        paddedHeight_, (batch.width() / 2) + 1, stream))
    {
        return false;
    }

    if (cufftExecC2R(planCache_.paganinInverse(),
        deviceSpectrum_, reinterpret_cast<cufftReal*>(devicePaddedReal_)) != CUFFT_SUCCESS)
    {
        return false;
    }

    return launchCropPaganinOutputY(devicePaddedReal_, batch.rawAngleMajor(),
        batch.currentAngleCount(), batch.height(), paddedHeight_, batch.width(), padTop_,
        static_cast<float>(paddedHeight_ * batch.width()), 1e-6f, stream);
}

} // namespace ct
