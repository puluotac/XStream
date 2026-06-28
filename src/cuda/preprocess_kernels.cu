#include "cuda/preprocess_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace
{

constexpr float kPiF = 3.14159265358979323846f;
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

__global__ void applyCorrectionKernel(float* rawAngleMajor, int angleCount, int height, int width,
    const float* flat, const float* dark, int correctionMode)
{
    const size_t pixelCount = static_cast<size_t>(height) * width;
    const size_t totalCount = static_cast<size_t>(angleCount) * pixelCount;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const size_t pixelIndex = index % pixelCount;
        const float raw = rawAngleMajor[index];

        if (correctionMode == 0)
        {
            rawAngleMajor[index] = raw / 65535.0f;
            continue;
        }

        const float flatValue = (flat != nullptr) ? flat[pixelIndex] : 65535.0f;
        const float darkValue = (dark != nullptr) ? dark[pixelIndex] : 0.0f;
        const float denominator = flatValue - darkValue + 1e-6f;
        rawAngleMajor[index] = (raw - darkValue) / denominator;
    }
}

__global__ void applyCorrectionRangeKernel(float* rawAngleMajor, int angleOffset, int angleCount,
    int height, int width, const float* flat, const float* dark, int correctionMode)
{
    const size_t pixelCount = static_cast<size_t>(height) * width;
    const size_t totalCount = static_cast<size_t>(angleCount) * pixelCount;
    for (size_t localIndex = linearThreadIndex(); localIndex < totalCount; localIndex += linearThreadStride())
    {
        const size_t pixelIndex = localIndex % pixelCount;
        const size_t index = (static_cast<size_t>(angleOffset) * pixelCount) + localIndex;
        const float raw = rawAngleMajor[index];

        if (correctionMode == 0)
        {
            rawAngleMajor[index] = raw / 65535.0f;
            continue;
        }

        const float flatValue = (flat != nullptr) ? flat[pixelIndex] : 65535.0f;
        const float darkValue = (dark != nullptr) ? dark[pixelIndex] : 0.0f;
        const float denominator = flatValue - darkValue + 1e-6f;
        rawAngleMajor[index] = (raw - darkValue) / denominator;
    }
}

__global__ void transposeAngleMajorToReconKernel(const float* rawAngleMajor, float* reconLayout,
    int angleCount, int height, int width)
{
    const size_t totalCount = static_cast<size_t>(angleCount) * height * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        const int tmp = static_cast<int>(index / width);
        const int y = tmp % height;
        const int angle = tmp / height;

        const size_t dstIndex = (static_cast<size_t>(y) * angleCount + angle) * width + x;
        reconLayout[dstIndex] = rawAngleMajor[index];
    }
}

__global__ void applyMethodAndTaperKernel(float* reconLayout, int height, int angleCount, int width,
    int preprocessMethod, float airValue, int taperWidth)
{
    const size_t totalCount = static_cast<size_t>(height) * angleCount * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        float value = reconLayout[index];

        if (preprocessMethod == 1)
        {
            value = -logf(fmaxf(value, 1e-6f));
        }
        else
        {
            value = airValue - value;
        }

        if (taperWidth > 0 && taperWidth < (width / 2))
        {
            float taper = 1.0f;
            if (x < taperWidth)
            {
                const float t = (taperWidth == 1) ? 0.0f : static_cast<float>(x) / static_cast<float>(taperWidth - 1);
                taper = sinf(t * kPiF * 0.5f);
            }
            else if (x >= (width - taperWidth))
            {
                const int edge = width - 1 - x;
                const float t = (taperWidth == 1) ? 0.0f : static_cast<float>(edge) / static_cast<float>(taperWidth - 1);
                taper = sinf(t * kPiF * 0.5f);
            }
            value *= taper;
        }

        reconLayout[index] = value;
    }
}

__device__ __forceinline__ int reflectIndex(int index, int size)
{
    if (size <= 1)
    {
        return 0;
    }

    while (index < 0 || index >= size)
    {
        if (index < 0)
        {
            index = -index;
        }
        else
        {
            index = (2 * size - 2) - index;
        }
    }

    return index;
}

__global__ void padPaganinInputReflectYKernel(const float* rawAngleMajor, float* paddedInput,
    int angleCount, int height, int paddedHeight, int width, int padTop)
{
    const size_t totalCount = static_cast<size_t>(angleCount) * paddedHeight * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        const int tmp = static_cast<int>(index / width);
        const int yPad = tmp % paddedHeight;
        const int angle = tmp / paddedHeight;
        const int sourceY = reflectIndex(yPad - padTop, height);

        const size_t srcIndex = (static_cast<size_t>(angle) * height + sourceY) * width + x;
        paddedInput[index] = rawAngleMajor[srcIndex];
    }
}

__global__ void multiplySpectrumByRealKernel(cufftComplex* spectrum, const float* filterKernel,
    int spectrumElementCount, int repeatedCount)
{
    const size_t totalCount = static_cast<size_t>(spectrumElementCount) * repeatedCount;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int filterIndex = static_cast<int>(index % spectrumElementCount);
        const float factor = filterKernel[filterIndex];
        spectrum[index].x *= factor;
        spectrum[index].y *= factor;
    }
}

__global__ void normalizeClampKernel(float* data, size_t elementCount, float scale, float clampMin)
{
    for (size_t index = linearThreadIndex(); index < elementCount; index += linearThreadStride())
    {
        data[index] = fmaxf(data[index] / scale, clampMin);
    }
}

__global__ void cropPaganinOutputYKernel(const float* paddedInput, float* rawAngleMajor,
    int angleCount, int height, int paddedHeight, int width, int padTop, float scale, float clampMin)
{
    const size_t totalCount = static_cast<size_t>(angleCount) * height * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        const int tmp = static_cast<int>(index / width);
        const int y = tmp % height;
        const int angle = tmp / height;
        const int yPad = y + padTop;
        const size_t srcIndex = (static_cast<size_t>(angle) * paddedHeight + yPad) * width + x;
        rawAngleMajor[index] = fmaxf(paddedInput[srcIndex] / scale, clampMin);
    }
}

__global__ void prepareFbpInputKernel(const float* reconLayout, float* paddedInput,
    int transformCount, int width, int paddedLength)
{
    const size_t totalCount = static_cast<size_t>(transformCount) * paddedLength;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % paddedLength);
        const int transform = static_cast<int>(index / paddedLength);
        paddedInput[index] = (x < width) ? reconLayout[static_cast<size_t>(transform) * width + x] : 0.0f;
    }
}

__global__ void extractFbpOutputKernel(const float* paddedOutput, float* reconLayout,
    int transformCount, int width, int paddedLength, float scale)
{
    const size_t totalCount = static_cast<size_t>(transformCount) * width;
    for (size_t index = linearThreadIndex(); index < totalCount; index += linearThreadStride())
    {
        const int x = static_cast<int>(index % width);
        const int transform = static_cast<int>(index / width);
        reconLayout[index] = paddedOutput[static_cast<size_t>(transform) * paddedLength + x] / scale;
    }
}

} // namespace

namespace ct
{

bool launchApplyCorrection(float* rawAngleMajor, int angleCount, int height, int width,
    const float* flat, const float* dark, int correctionMode, cudaStream_t stream)
{
    cudaGetLastError();
    const int angleChunk = 1;
    const size_t pixelsPerAngle = static_cast<size_t>(height) * width;
    for (int angleOffset = 0; angleOffset < angleCount; angleOffset += angleChunk)
    {
        const int currentAngles = std::min(angleChunk, angleCount - angleOffset);
        const size_t totalCount = static_cast<size_t>(currentAngles) * pixelsPerAngle;
        applyCorrectionRangeKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
            rawAngleMajor, angleOffset, currentAngles, height, width, flat, dark, correctionMode);
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return false;
        }
    }
    return true;
}

bool launchApplyCorrectionRange(float* rawAngleMajor, int angleOffset, int angleCount, int height, int width,
    const float* flat, const float* dark, int correctionMode, cudaStream_t stream)
{
    if (angleOffset < 0 || angleCount <= 0)
    {
        return false;
    }
    cudaGetLastError();
    const size_t totalCount = static_cast<size_t>(angleCount) * height * width;
    applyCorrectionRangeKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        rawAngleMajor, angleOffset, angleCount, height, width, flat, dark, correctionMode);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchTransposeAngleMajorToRecon(const float* rawAngleMajor, float* reconLayout,
    int angleCount, int height, int width, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(angleCount) * height * width;
    cudaGetLastError();
    transposeAngleMajorToReconKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        rawAngleMajor, reconLayout, angleCount, height, width);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchApplyMethodAndTaper(float* reconLayout, int height, int angleCount, int width,
    int preprocessMethod, float airValue, int taperWidth, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(height) * angleCount * width;
    cudaGetLastError();
    applyMethodAndTaperKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        reconLayout, height, angleCount, width, preprocessMethod, airValue, taperWidth);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchPadPaganinInputReflectY(const float* rawAngleMajor, float* paddedInput,
    int angleCount, int height, int paddedHeight, int width, int padTop, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(angleCount) * paddedHeight * width;
    cudaGetLastError();
    padPaganinInputReflectYKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        rawAngleMajor, paddedInput, angleCount, height, paddedHeight, width, padTop);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchMultiplyPaganinSpectrum(cufftComplex* spectrum, const float* filterKernel,
    int batchAngles, int paddedHeight, int spectrumWidth, cudaStream_t stream)
{
    const int spectrumElementCount = paddedHeight * spectrumWidth;
    const size_t totalCount = static_cast<size_t>(batchAngles) * spectrumElementCount;
    cudaGetLastError();
    multiplySpectrumByRealKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        spectrum, filterKernel, spectrumElementCount, batchAngles);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchNormalizeClamp(float* data, size_t elementCount, float scale, float clampMin, cudaStream_t stream)
{
    cudaGetLastError();
    normalizeClampKernel<<<makeGrid(elementCount), kKernelBlockSize, 0, stream>>>(data, elementCount, scale, clampMin);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchCropPaganinOutputY(const float* paddedInput, float* rawAngleMajor,
    int angleCount, int height, int paddedHeight, int width, int padTop, float scale,
    float clampMin, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(angleCount) * height * width;
    cudaGetLastError();
    cropPaganinOutputYKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        paddedInput, rawAngleMajor, angleCount, height, paddedHeight, width, padTop, scale, clampMin);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchPrepareFbpInput(const float* reconLayout, float* paddedInput,
    int transformCount, int width, int paddedLength, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(transformCount) * paddedLength;
    cudaGetLastError();
    prepareFbpInputKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        reconLayout, paddedInput, transformCount, width, paddedLength);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchMultiplyFbpSpectrum(cufftComplex* spectrum, const float* filterSpectrum,
    int transformCount, int spectrumLength, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(transformCount) * spectrumLength;
    cudaGetLastError();
    multiplySpectrumByRealKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        spectrum, filterSpectrum, spectrumLength, transformCount);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchExtractFbpOutput(const float* paddedOutput, float* reconLayout,
    int transformCount, int width, int paddedLength, float scale, cudaStream_t stream)
{
    const size_t totalCount = static_cast<size_t>(transformCount) * width;
    cudaGetLastError();
    extractFbpOutputKernel<<<makeGrid(totalCount), kKernelBlockSize, 0, stream>>>(
        paddedOutput, reconLayout, transformCount, width, paddedLength, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace ct
