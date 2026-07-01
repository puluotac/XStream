#include "cuda/stripe_removal_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace
{

constexpr int kKernelBlockSize = 256;
constexpr int kMaxGridX = 65535;

dim3 makeGrid(size_t elementCount)
{
    const size_t blockCount = (elementCount + kKernelBlockSize - 1) / kKernelBlockSize;
    const unsigned int gridX = static_cast<unsigned int>(std::min<size_t>(blockCount, kMaxGridX));
    const unsigned int gridY = static_cast<unsigned int>((blockCount + kMaxGridX - 1) / kMaxGridX);
    return dim3(std::max(1u, gridX), std::max(1u, gridY), 1u);
}

__device__ __forceinline__ size_t linearThreadIndex()
{
    return (static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ size_t linearThreadStride()
{
    return static_cast<size_t>(gridDim.x) * gridDim.y * blockDim.x;
}

__device__ __forceinline__ int symmetricIndex(int idx, int n)
{
    if (n <= 1)
    {
        return 0;
    }
    while (idx < 0 || idx >= n)
    {
        if (idx < 0)
        {
            idx = -idx - 1;
        }
        if (idx >= n)
        {
            idx = 2 * n - 1 - idx;
        }
    }
    return idx;
}

__device__ __forceinline__ int reflectIndex(int idx, int n)
{
    if (n <= 1)
    {
        return 0;
    }
    while (idx < 0 || idx >= n)
    {
        if (idx < 0)
        {
            idx = -idx;
        }
        if (idx >= n)
        {
            idx = 2 * n - 2 - idx;
        }
    }
    return idx;
}

__device__ __forceinline__ int rightSymmetricIndex(int idx, int n)
{
    if (n <= 1)
    {
        return 0;
    }
    while (idx >= n)
    {
        idx = 2 * n - 1 - idx;
        if (idx < 0)
        {
            idx = -idx - 1;
        }
    }
    return idx;
}

__device__ __forceinline__ float db5DecLo(int i)
{
    constexpr float coeff[10] = {
        0.160102397974f, 0.603829269797f, 0.724308528438f, 0.138428145901f, -0.242294887066f,
        -0.032244869585f, 0.07757149384f, -0.006241490213f, -0.012580751999f, 0.003335725285f
    };
    return coeff[i];
}

__device__ __forceinline__ float db5DecHi(int i)
{
    constexpr float coeff[10] = {
        0.003335725285f, 0.012580751999f, -0.006241490213f, -0.07757149384f, -0.032244869585f,
        0.242294887066f, 0.138428145901f, -0.724308528438f, 0.603829269797f, -0.160102397974f
    };
    return coeff[i];
}

__device__ __forceinline__ float db5RecLo(int i)
{
    constexpr float coeff[10] = {
        0.003335725285f, -0.012580751999f, -0.006241490213f, 0.07757149384f, -0.032244869585f,
        -0.242294887066f, 0.138428145901f, 0.724308528438f, 0.603829269797f, 0.160102397974f
    };
    return coeff[i];
}

__device__ __forceinline__ float db5RecHi(int i)
{
    constexpr float coeff[10] = {
        -0.160102397974f, 0.603829269797f, -0.724308528438f, 0.138428145901f, 0.242294887066f,
        -0.032244869585f, -0.07757149384f, -0.006241490213f, 0.012580751999f, 0.003335725285f
    };
    return coeff[i];
}

__global__ void packSinogramColumnsKernel(const float* sino, float* packed,
    int zCount, int angleCount, int width)
{
    const size_t totalCount = static_cast<size_t>(zCount) * angleCount * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int angle = static_cast<int>(index % angleCount);
        const int vector = static_cast<int>(index / angleCount);
        const int x = vector % width;
        const int z = vector / width;
        packed[index] = sino[(static_cast<size_t>(z) * angleCount + angle) * width + x];
    }
}

__global__ void unpackSinogramColumnsKernel(const float* packed, float* sino,
    int zCount, int angleCount, int width)
{
    const size_t totalCount = static_cast<size_t>(zCount) * angleCount * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int angle = static_cast<int>(index % angleCount);
        const int vector = static_cast<int>(index / angleCount);
        const int x = vector % width;
        const int z = vector / width;
        sino[(static_cast<size_t>(z) * angleCount + angle) * width + x] =
            packed[index] / static_cast<float>(angleCount);
    }
}

__global__ void padStripeInputKernel(const float* sino, float* padded,
    int zCount, int angleCount, int paddedAngleCount, int width, int angleShift)
{
    const size_t totalCount = static_cast<size_t>(zCount) * paddedAngleCount * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        const int paddedAngle = static_cast<int>((index / width) % paddedAngleCount);
        const int z = static_cast<int>(index / (static_cast<size_t>(paddedAngleCount) * width));
        float value = 0.0f;
        const int sourceAngle = paddedAngle - angleShift;
        if (sourceAngle >= 0 && sourceAngle < angleCount)
        {
            value = sino[(static_cast<size_t>(z) * angleCount + sourceAngle) * width + x];
        }
        padded[index] = value;
    }
}

__global__ void cropStripeOutputKernel(const float* padded, float* sino,
    int zCount, int angleCount, int paddedAngleCount, int width, int angleShift)
{
    const size_t totalCount = static_cast<size_t>(zCount) * angleCount * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        const int angle = static_cast<int>((index / width) % angleCount);
        const int z = static_cast<int>(index / (static_cast<size_t>(angleCount) * width));
        sino[index] = padded[(static_cast<size_t>(z) * paddedAngleCount + angle + angleShift) * width + x];
    }
}

__global__ void dwtAxis1Db5Kernel(const float* src, float* approx, float* detail,
    int zCount, int dataLen, int width, int outLen, int paddedLen)
{
    const size_t totalCount = static_cast<size_t>(zCount) * outLen * width;
    constexpr int kFiltLen = 10;
    constexpr int kPadLeft = 8;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        const int outPos = static_cast<int>((index / width) % outLen);
        const int z = static_cast<int>(index / (static_cast<size_t>(outLen) * width));
        float lo = 0.0f;
        float hi = 0.0f;
        const int base = outPos * 2 - kPadLeft;
        for (int j = 0; j < kFiltLen; ++j)
        {
            int pos = symmetricIndex(base + j, paddedLen);
            pos = rightSymmetricIndex(pos, dataLen);
            const float v = src[(static_cast<size_t>(z) * dataLen + pos) * width + x];
            lo += v * db5DecLo(j);
            hi += v * db5DecHi(j);
        }
        approx[index] = lo;
        detail[index] = hi;
    }
}

__global__ void dwtAxis2Db5Kernel(const float* src, float* approx, float* detail,
    int zCount, int height, int dataLen, int outLen, int paddedLen)
{
    const size_t totalCount = static_cast<size_t>(zCount) * height * outLen;
    constexpr int kFiltLen = 10;
    constexpr int kPadLeft = 8;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int outPos = static_cast<int>(index % outLen);
        const int y = static_cast<int>((index / outLen) % height);
        const int z = static_cast<int>(index / (static_cast<size_t>(height) * outLen));
        float lo = 0.0f;
        float hi = 0.0f;
        const int base = outPos * 2 - kPadLeft;
        const size_t rowBase = (static_cast<size_t>(z) * height + y) * dataLen;
        for (int j = 0; j < kFiltLen; ++j)
        {
            int pos = symmetricIndex(base + j, paddedLen);
            pos = rightSymmetricIndex(pos, dataLen);
            const float v = src[rowBase + pos];
            lo += v * db5DecLo(j);
            hi += v * db5DecHi(j);
        }
        approx[index] = lo;
        detail[index] = hi;
    }
}

__global__ void idwtAxis1Db5Kernel(const float* approx, const float* detail, float* out,
    int zCount, int coeffLen, int width, int targetLen)
{
    const size_t totalCount = static_cast<size_t>(zCount) * targetLen * width;
    constexpr int kFiltLen = 10;
    constexpr int kPadLeft = 1;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        const int t = static_cast<int>((index / width) % targetLen);
        const int z = static_cast<int>(index / (static_cast<size_t>(targetLen) * width));
        const int currLen = coeffLen * 2;
        const int start = ((currLen - targetLen) / 2) - 4;
        int outPos = start + t;
        outPos = min(max(outPos, 0), currLen - 1);

        float sum = 0.0f;
        for (int j = 0; j < kFiltLen; ++j)
        {
            const int pos = reflectIndex(outPos + j - kPadLeft, currLen);
            if ((pos & 1) == 0)
            {
                const int srcPos = pos >> 1;
                const size_t base = (static_cast<size_t>(z) * coeffLen + srcPos) * width + x;
                sum += approx[base] * db5RecLo(j) + detail[base] * db5RecHi(j);
            }
        }
        out[index] = sum;
    }
}

__global__ void idwtAxis2Db5Kernel(const float* approx, const float* detail, float* out,
    int zCount, int height, int coeffLen, int targetLen)
{
    const size_t totalCount = static_cast<size_t>(zCount) * height * targetLen;
    constexpr int kFiltLen = 10;
    constexpr int kPadLeft = 1;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int t = static_cast<int>(index % targetLen);
        const int y = static_cast<int>((index / targetLen) % height);
        const int z = static_cast<int>(index / (static_cast<size_t>(height) * targetLen));
        const int currLen = coeffLen * 2;
        const int start = ((currLen - targetLen) / 2) - 4;
        int outPos = start + t;
        outPos = min(max(outPos, 0), currLen - 1);

        float sum = 0.0f;
        const size_t rowBase = (static_cast<size_t>(z) * height + y) * coeffLen;
        for (int j = 0; j < kFiltLen; ++j)
        {
            const int pos = reflectIndex(outPos + j - kPadLeft, currLen);
            if ((pos & 1) == 0)
            {
                const int srcPos = pos >> 1;
                sum += approx[rowBase + srcPos] * db5RecLo(j) + detail[rowBase + srcPos] * db5RecHi(j);
            }
        }
        out[index] = sum;
    }
}

__global__ void dampStripeSpectrumKernel(cufftComplex* spectrum,
    int vectorCount, int spectrumLength, int angleCount, float sigma)
{
    const size_t totalCount = static_cast<size_t>(vectorCount) * spectrumLength;
    const float sigmaSafe = fmaxf(sigma, 0.01f);
    const float sigmaSq2 = 2.0f * sigmaSafe * sigmaSafe;
    const int shiftedCenter = angleCount / 2;

    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int freq = static_cast<int>(index % spectrumLength);
        if (freq == 0)
        {
            continue;
        }

        const int shifted = (freq + shiftedCenter) % angleCount;
        const float yHat = (static_cast<float>(shifted * 2 - angleCount) + 1.0f) * 0.5f;
        const float damp = -expm1f(-(yHat * yHat) / sigmaSq2);
        cufftComplex value = spectrum[index];
        value.x *= damp;
        value.y *= damp;
        spectrum[index] = value;
    }
}

} // namespace

namespace ct
{

bool launchPackSinogramColumnsForStripe(const float* sino, float* packed,
    int zCount, int angleCount, int width, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(zCount) * angleCount * width;
    cudaGetLastError();
    packSinogramColumnsKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        sino, packed, zCount, angleCount, width);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchPadStripeInput(const float* sino, float* padded,
    int zCount, int angleCount, int paddedAngleCount, int width, int angleShift, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(zCount) * paddedAngleCount * width;
    cudaGetLastError();
    padStripeInputKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        sino, padded, zCount, angleCount, paddedAngleCount, width, angleShift);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchDampStripeSpectrum(cufftComplex* spectrum,
    int vectorCount, int spectrumLength, int angleCount, float sigma, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(vectorCount) * spectrumLength;
    cudaGetLastError();
    dampStripeSpectrumKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        spectrum, vectorCount, spectrumLength, angleCount, sigma);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchUnpackSinogramColumnsForStripe(const float* packed, float* sino,
    int zCount, int angleCount, int width, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(zCount) * angleCount * width;
    cudaGetLastError();
    unpackSinogramColumnsKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        packed, sino, zCount, angleCount, width);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchCropStripeOutput(const float* padded, float* sino,
    int zCount, int angleCount, int paddedAngleCount, int width, int angleShift, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(zCount) * angleCount * width;
    cudaGetLastError();
    cropStripeOutputKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        padded, sino, zCount, angleCount, paddedAngleCount, width, angleShift);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchDwtAxis1Db5(const float* src, float* approx, float* detail,
    int zCount, int dataLen, int width, int outLen, int paddedLen, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(zCount) * outLen * width;
    cudaGetLastError();
    dwtAxis1Db5Kernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        src, approx, detail, zCount, dataLen, width, outLen, paddedLen);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchDwtAxis2Db5(const float* src, float* approx, float* detail,
    int zCount, int height, int dataLen, int outLen, int paddedLen, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(zCount) * height * outLen;
    cudaGetLastError();
    dwtAxis2Db5Kernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        src, approx, detail, zCount, height, dataLen, outLen, paddedLen);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchIdwtAxis1Db5(const float* approx, const float* detail, float* out,
    int zCount, int coeffLen, int width, int targetLen, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(zCount) * targetLen * width;
    cudaGetLastError();
    idwtAxis1Db5Kernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        approx, detail, out, zCount, coeffLen, width, targetLen);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchIdwtAxis2Db5(const float* approx, const float* detail, float* out,
    int zCount, int height, int coeffLen, int targetLen, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(zCount) * height * targetLen;
    cudaGetLastError();
    idwtAxis2Db5Kernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        approx, detail, out, zCount, height, coeffLen, targetLen);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace ct
