#include "cuda/offline_tomo_kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace
{

constexpr float kPiF = 3.14159265358979323846f;

__global__ void offlineFbpBackprojectionKernel(
    const float* __restrict__ sino,
    const float2* __restrict__ geom,
    int angleCount,
    int volW,
    int zChunkHeight,
    float corShift,
    float* __restrict__ vol)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= volW || y >= volW || z >= zChunkHeight)
    {
        return;
    }

    const float cx = static_cast<float>(x) - static_cast<float>(volW - 1) * 0.5f;
    const float cy = static_cast<float>(y) - static_cast<float>(volW - 1) * 0.5f;
    const float detOffset = static_cast<float>(volW - 1) * 0.5f;
    const float* sinoZ = sino + static_cast<size_t>(z) * angleCount * volW;

    float sum = 0.0f;
    for (int a = 0; a < angleCount; ++a)
    {
        const float2 g = geom[a];
        const float u = cx * g.x + cy * g.y + corShift + detOffset;
        const int u0 = static_cast<int>(floorf(u));
        if (u0 >= 0 && u0 < volW - 1)
        {
            const float* row = sinoZ + static_cast<size_t>(a) * volW;
            sum += row[u0] + (u - static_cast<float>(u0)) * (row[u0 + 1] - row[u0]);
        }
    }

    vol[(static_cast<size_t>(z) * volW + y) * volW + x] = sum;
}

__device__ void atomicAddComplex(cufftComplex* addr, cufftComplex val)
{
    atomicAdd(&addr->x, val.x);
    atomicAdd(&addr->y, val.y);
}

__device__ float i0Approx(float x)
{
    const float t = x * x / 14.0625f;
    return 1.0f + t * (3.5156229f + t * (3.0899424f + t * (1.2067492f
        + t * (0.2659732f + t * (0.0360768f + t * 0.0045813f)))));
}

__device__ float kbWeight(float dist, float width, float beta)
{
    if (dist >= width * 0.5f)
    {
        return 0.0f;
    }
    const float arg = beta * sqrtf(fmaxf(0.0f, 1.0f - powf(2.0f * dist / width, 2.0f)));
    return i0Approx(arg) / i0Approx(beta);
}

__device__ cufftComplex complexMul(cufftComplex a, cufftComplex b)
{
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ cufftComplex complexConj(cufftComplex a)
{
    return make_cuFloatComplex(a.x, -a.y);
}

__global__ void gridrecAccumulateKernel(
    const cufftComplex* __restrict__ sinoFreq,
    const double* __restrict__ vectors,
    int angleCount,
    int zChunkHeight,
    int freqWidth,
    int gridSize,
    int paddedWidth,
    cufftComplex* __restrict__ gridVol)
{
    const int kIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int aIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (kIdx >= freqWidth || aIdx >= angleCount)
    {
        return;
    }

    const float ux = static_cast<float>(vectors[aIdx * 12 + 6]);
    const float uy = static_cast<float>(vectors[aIdx * 12 + 7]);
    const float centerPos = static_cast<float>(gridSize - 1) * 0.5f;
    const float corShift = static_cast<float>(vectors[aIdx * 12 + 3] * vectors[aIdx * 12 + 6]
        + vectors[aIdx * 12 + 4] * vectors[aIdx * 12 + 7]);
    const float totalShift = corShift + centerPos;

    const float k = static_cast<float>(kIdx);
    const float scale = static_cast<float>(gridSize) / static_cast<float>(paddedWidth);
    const float uCenter = k * ux * scale;
    const float vCenter = k * uy * scale;
    const float cx = static_cast<float>(gridSize) * 0.5f;
    const float cy = static_cast<float>(gridSize) * 0.5f;
    const float kernWidth = 2.0f;
    const float beta = 6.0f;

    int iul = max(0, static_cast<int>(ceilf(uCenter + cx - kernWidth)));
    int iuh = min(gridSize - 1, static_cast<int>(floorf(uCenter + cx + kernWidth)));
    int ivl = max(0, static_cast<int>(ceilf(vCenter + cy - kernWidth)));
    int ivh = min(gridSize - 1, static_cast<int>(floorf(vCenter + cy + kernWidth)));

    const float phaseAngle = 2.0f * kPiF * k * totalShift / static_cast<float>(paddedWidth);
    const cufftComplex phaseShift = make_cuFloatComplex(cosf(phaseAngle), sinf(phaseAngle));

    for (int z = 0; z < zChunkHeight; ++z)
    {
        const size_t sinoIndex = (static_cast<size_t>(z) * angleCount + aIdx) * freqWidth + kIdx;
        cufftComplex rawValue = sinoFreq[sinoIndex];
        if ((rawValue.x * rawValue.x + rawValue.y * rawValue.y) < 1e-18f)
        {
            continue;
        }

        const cufftComplex value = complexMul(rawValue, phaseShift);
        for (int iu = iul; iu <= iuh; ++iu)
        {
            const float wu = kbWeight(fabsf(static_cast<float>(iu) - cx - uCenter), kernWidth * 2.0f, beta);
            for (int iv = ivl; iv <= ivh; ++iv)
            {
                const float wv = kbWeight(fabsf(static_cast<float>(iv) - cy - vCenter), kernWidth * 2.0f, beta);
                cufftComplex weighted = make_cuFloatComplex(value.x * wu * wv, value.y * wu * wv);
                const size_t gridIndex = (static_cast<size_t>(z) * gridSize + iv) * gridSize + iu;
                atomicAddComplex(&gridVol[gridIndex], weighted);
            }
        }

        if (kIdx > 0 && kIdx < freqWidth - 1)
        {
            const float uNeg = -uCenter;
            const float vNeg = -vCenter;
            const int iulN = max(0, static_cast<int>(ceilf(uNeg + cx - kernWidth)));
            const int iuhN = min(gridSize - 1, static_cast<int>(floorf(uNeg + cx + kernWidth)));
            const int ivlN = max(0, static_cast<int>(ceilf(vNeg + cy - kernWidth)));
            const int ivhN = min(gridSize - 1, static_cast<int>(floorf(vNeg + cy + kernWidth)));
            const cufftComplex valueConj = complexConj(value);
            for (int iu = iulN; iu <= iuhN; ++iu)
            {
                const float wu = kbWeight(fabsf(static_cast<float>(iu) - cx - uNeg), kernWidth * 2.0f, beta);
                for (int iv = ivlN; iv <= ivhN; ++iv)
                {
                    const float wv = kbWeight(fabsf(static_cast<float>(iv) - cy - vNeg), kernWidth * 2.0f, beta);
                    cufftComplex weighted = make_cuFloatComplex(valueConj.x * wu * wv, valueConj.y * wu * wv);
                    const size_t gridIndex = (static_cast<size_t>(z) * gridSize + iv) * gridSize + iu;
                    atomicAddComplex(&gridVol[gridIndex], weighted);
                }
            }
        }
    }
}

__global__ void gridrecIfftshiftCopyKernel(const cufftComplex* __restrict__ src,
    int zChunkHeight, int gridH, int gridW, cufftComplex* __restrict__ dst)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(zChunkHeight) * gridH * gridW;
    if (index >= total)
    {
        return;
    }

    const int x = static_cast<int>(index % gridW);
    const size_t tmp = index / gridW;
    const int y = static_cast<int>(tmp % gridH);
    const int z = static_cast<int>(tmp / gridH);
    const int srcY = (y + gridH / 2) % gridH;
    const int srcX = (x + gridW / 2) % gridW;
    dst[index] = src[(static_cast<size_t>(z) * gridH + srcY) * gridW + srcX];
}

__global__ void gridrecFinalizeCropKernel(const cufftComplex* __restrict__ imgComplex,
    const float* __restrict__ deapodMap, int zChunkHeight, int gridH, int gridW,
    int outH, int outW, int cropStart, float scalingFactor, float* __restrict__ outVol)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (z >= zChunkHeight || y >= outH || x >= outW)
    {
        return;
    }

    const int centeredY = y + cropStart;
    const int centeredX = x + cropStart;
    const int srcY = (centeredY + gridH / 2) % gridH;
    const int srcX = (centeredX + gridW / 2) % gridW;
    const size_t srcIndex = (static_cast<size_t>(z) * gridH + srcY) * gridW + srcX;
    const size_t deapodIndex = static_cast<size_t>(centeredY) * gridW + centeredX;
    const size_t outIndex = (static_cast<size_t>(z) * outH + y) * outW + x;
    const float denom = fmaxf(deapodMap[deapodIndex], 1e-6f);
    outVol[outIndex] = imgComplex[srcIndex].x * scalingFactor / denom;
}

__global__ void convertFloatVolumeToUint16Kernel(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    size_t voxelCount,
    float vmin,
    float invRange)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= voxelCount)
    {
        return;
    }

    const float value = src[index];
    if (!isfinite(value) || value <= vmin)
    {
        dst[index] = 0;
        return;
    }
    const float normalized = (value - vmin) * invRange;
    if (normalized >= 1.0f)
    {
        dst[index] = 65535;
        return;
    }
    dst[index] = static_cast<uint16_t>(normalized * 65535.0f + 0.5f);
}

} // namespace

namespace ct
{

void launchOfflineFbpBackprojection(const float* sinoSpatial, const float2* geom,
    int angleCount, int volW, int zChunkHeight, float corShift, float* outVolume,
    cudaStream_t stream)
{
    const dim3 block(16, 8, 2);
    const dim3 grid((volW + block.x - 1) / block.x,
        (volW + block.y - 1) / block.y,
        (zChunkHeight + block.z - 1) / block.z);
    offlineFbpBackprojectionKernel<<<grid, block, 0, stream>>>(
        sinoSpatial, geom, angleCount, volW, zChunkHeight, corShift, outVolume);
}

void launchGridrecAccumulate(const cufftComplex* sinoFreq, const double* vectors,
    int angleCount, int zChunkHeight, int freqWidth, int gridSize, int paddedWidth,
    cufftComplex* gridVolume, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((freqWidth + block.x - 1) / block.x,
        (angleCount + block.y - 1) / block.y);
    gridrecAccumulateKernel<<<grid, block, 0, stream>>>(
        sinoFreq, vectors, angleCount, zChunkHeight, freqWidth, gridSize, paddedWidth, gridVolume);
}

void launchGridrecIfftshiftCopy(const cufftComplex* src, int zChunkHeight,
    int gridH, int gridW, cufftComplex* dst, cudaStream_t stream)
{
    const int block = 256;
    const size_t total = static_cast<size_t>(zChunkHeight) * gridH * gridW;
    const dim3 grid(static_cast<unsigned int>((total + block - 1) / block));
    gridrecIfftshiftCopyKernel<<<grid, block, 0, stream>>>(src, zChunkHeight, gridH, gridW, dst);
}

void launchGridrecFinalizeCrop(const cufftComplex* imgComplex, const float* deapodMap,
    int zChunkHeight, int gridH, int gridW, int outH, int outW, int cropStart,
    float scalingFactor, float* outVolume, cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((outW + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y,
        zChunkHeight);
    gridrecFinalizeCropKernel<<<grid, block, 0, stream>>>(
        imgComplex, deapodMap, zChunkHeight, gridH, gridW, outH, outW, cropStart, scalingFactor, outVolume);
}

void launchConvertFloatVolumeToUint16(const float* src, uint16_t* dst,
    size_t voxelCount, float vmin, float vmax, cudaStream_t stream)
{
    const int block = 256;
    const float range = fmaxf(vmax - vmin, 1.0e-6f);
    const float invRange = 1.0f / range;
    const dim3 grid(static_cast<unsigned int>((voxelCount + block - 1) / block));
    convertFloatVolumeToUint16Kernel<<<grid, block, 0, stream>>>(src, dst, voxelCount, vmin, invRange);
}

} // namespace ct
