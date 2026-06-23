#include <cuda_runtime.h>

namespace
{

constexpr int kBpVectorTile = 64;
constexpr int kBpZTile = 4;
constexpr int kBpConstantVectorLimit = 64;

__constant__ float4 cBpPackedVectors[kBpConstantVectorLimit];

__device__ __forceinline__ float sampleSinogramLinear(
    const float* sino, float u, int rowIndex, int detWidth)
{
    if (u < 0.0f || u >= static_cast<float>(detWidth - 1))
    {
        return 0.0f;
    }

    const int u0 = static_cast<int>(floorf(u));
    const float* row = sino + static_cast<size_t>(rowIndex) * detWidth;
    return row[u0] + (u - static_cast<float>(u0)) * (row[u0 + 1] - row[u0]);
}

template <bool UseConstantVectors>
__global__ void bp3d_optimized(
    const float* __restrict__ sino,
    const float4* __restrict__ packedVectors,
    const int currentBatchSize,
    const int volW,
    const int hOffset,
    const int batchH,
    float* __restrict__ volume)
{
    __shared__ float4 sVec[kBpVectorTile];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zBase = static_cast<int>(blockIdx.z) * kBpZTile;
    const bool active = x < volW && y < volW && zBase < batchH;

    const float cx = static_cast<float>(x) - (volW - 1) * 0.5f;
    const float cy = static_cast<float>(y) - (volW - 1) * 0.5f;
    const float detOffset = (volW - 1) * 0.5f;

    float accum[kBpZTile] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int aBase = 0; aBase < currentBatchSize; aBase += kBpVectorTile)
    {
        const int aLimit = min(kBpVectorTile, currentBatchSize - aBase);

        if constexpr (!UseConstantVectors)
        {
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;
            const int blockSize = blockDim.x * blockDim.y;
            for (int i = tid; i < aLimit; i += blockSize)
            {
                sVec[i] = packedVectors[aBase + i];
            }
        }

        __syncthreads();

        if (active)
        {
            for (int a = 0; a < aLimit; ++a)
            {
                const float4 vec = UseConstantVectors ? cBpPackedVectors[aBase + a] : sVec[a];
                const float u = (cx + vec.x) * vec.z + (cy + vec.y) * vec.w + detOffset;

                if (u < 0.0f || u >= static_cast<float>(volW - 1))
                {
                    continue;
                }

                const int angleIndex = aBase + a;
#pragma unroll
                for (int zi = 0; zi < kBpZTile; ++zi)
                {
                    const int localZ = zBase + zi;
                    if (localZ < batchH)
                    {
                        const int rowIndex = localZ * currentBatchSize + angleIndex;
                        accum[zi] += sampleSinogramLinear(sino, u, rowIndex, volW);
                    }
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int zi = 0; zi < kBpZTile; ++zi)
    {
        const int localZ = zBase + zi;
        if (!active || localZ >= batchH)
        {
            continue;
        }

        const size_t globalIndex = static_cast<size_t>(localZ + hOffset) * volW * volW
            + static_cast<size_t>(y) * volW + x;
        volume[globalIndex] += accum[zi];
    }
}

template <bool UseConstantVectors>
__global__ void bp3d_slab_optimized(
    const float* __restrict__ sino,
    const float4* __restrict__ packedVectors,
    const int currentBatchSize,
    const int volW,
    const int globalZOffset,
    const int slabH,
    float* __restrict__ slabVolume)
{
    __shared__ float4 sVec[kBpVectorTile];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zBase = static_cast<int>(blockIdx.z) * kBpZTile;
    const bool active = x < volW && y < volW && zBase < slabH;

    const float cx = static_cast<float>(x) - (volW - 1) * 0.5f;
    const float cy = static_cast<float>(y) - (volW - 1) * 0.5f;
    const float detOffset = (volW - 1) * 0.5f;

    float accum[kBpZTile] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int aBase = 0; aBase < currentBatchSize; aBase += kBpVectorTile)
    {
        const int aLimit = min(kBpVectorTile, currentBatchSize - aBase);

        if constexpr (!UseConstantVectors)
        {
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;
            const int blockSize = blockDim.x * blockDim.y;
            for (int i = tid; i < aLimit; i += blockSize)
            {
                sVec[i] = packedVectors[aBase + i];
            }
        }

        __syncthreads();

        if (active)
        {
            for (int a = 0; a < aLimit; ++a)
            {
                const float4 vec = UseConstantVectors ? cBpPackedVectors[aBase + a] : sVec[a];
                const float u = (cx + vec.x) * vec.z + (cy + vec.y) * vec.w + detOffset;

                if (u < 0.0f || u >= static_cast<float>(volW - 1))
                {
                    continue;
                }

                const int angleIndex = aBase + a;
#pragma unroll
                for (int zi = 0; zi < kBpZTile; ++zi)
                {
                    const int localZ = zBase + zi;
                    if (localZ < slabH)
                    {
                        const int rowIndex = localZ * currentBatchSize + angleIndex;
                        accum[zi] += sampleSinogramLinear(sino, u, rowIndex, volW);
                    }
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int zi = 0; zi < kBpZTile; ++zi)
    {
        const int localZ = zBase + zi;
        if (!active || localZ >= slabH)
        {
            continue;
        }

        const size_t slabIndex = static_cast<size_t>(localZ) * volW * volW
            + static_cast<size_t>(y) * volW + x;
        slabVolume[slabIndex] += accum[zi];
    }
}

} // namespace

namespace ct
{

bool uploadBpPackedVectorsToConstantAsync(const float4* packedVectors, int angleCount, cudaStream_t stream)
{
    if (packedVectors == nullptr || angleCount <= 0 || angleCount > kBpConstantVectorLimit)
    {
        return false;
    }

    return cudaMemcpyToSymbolAsync(cBpPackedVectors, packedVectors,
        static_cast<size_t>(angleCount) * sizeof(float4), 0,
        cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
}

void launchBp3dOptimizedKernel(const float* sino, const float4* packedVectors,
    bool useConstantVectors, int currentBatchSize, int volW, int hOffset,
    int batchH, float* volume, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((volW + block.x - 1) / block.x,
        (volW + block.y - 1) / block.y,
        (batchH + kBpZTile - 1) / kBpZTile);

    if (useConstantVectors)
    {
        bp3d_optimized<true><<<grid, block, 0, stream>>>(
            sino, packedVectors, currentBatchSize, volW, hOffset, batchH, volume);
    }
    else
    {
        bp3d_optimized<false><<<grid, block, 0, stream>>>(
            sino, packedVectors, currentBatchSize, volW, hOffset, batchH, volume);
    }
}

void launchBp3dSlabOptimizedKernel(const float* sino, const float4* packedVectors,
    bool useConstantVectors, int currentBatchSize, int volW, int globalZOffset,
    int slabH, float* slabVolume, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((volW + block.x - 1) / block.x,
        (volW + block.y - 1) / block.y,
        (slabH + kBpZTile - 1) / kBpZTile);

    if (useConstantVectors)
    {
        bp3d_slab_optimized<true><<<grid, block, 0, stream>>>(
            sino, packedVectors, currentBatchSize, volW, globalZOffset, slabH, slabVolume);
    }
    else
    {
        bp3d_slab_optimized<false><<<grid, block, 0, stream>>>(
            sino, packedVectors, currentBatchSize, volW, globalZOffset, slabH, slabVolume);
    }
}

} // namespace ct
