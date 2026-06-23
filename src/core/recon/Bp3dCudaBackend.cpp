#include "core/recon/Bp3dCudaBackend.h"

#include "core/recon/GpuVectorBuffer.h"
#include "core/recon/GpuVolumeBuffer.h"

#include <cuda_runtime.h>

namespace ct
{

bool uploadBpPackedVectorsToConstantAsync(const float4* packedVectors, int angleCount, cudaStream_t stream);
void launchBp3dOptimizedKernel(const float* sino, const float4* packedVectors,
    bool useConstantVectors, int currentBatchSize, int volW, int hOffset,
    int batchH, float* volume, cudaStream_t stream);
void launchBp3dSlabOptimizedKernel(const float* sino, const float4* packedVectors,
    bool useConstantVectors, int currentBatchSize, int volW, int globalZOffset,
    int slabH, float* slabVolume, cudaStream_t stream);

bool Bp3dCudaBackend::initialize(const GeometryConfig& geometry, const ComputeConfig& compute)
{
    volumeWidth_ = geometry.volW;
    if (cudaSetDevice(compute.gpuIndex) != cudaSuccess)
    {
        return false;
    }

    return true;
}

bool Bp3dCudaBackend::backProjectChunk(const DeviceChunkView& chunk, const GpuVectorBuffer& vectors,
    GpuVolumeBuffer& volume, cudaStream_t stream) const
{
    if (chunk.deviceData == nullptr || chunk.chunkHeight <= 0 || chunk.angleCount <= 0 || chunk.detWidth <= 0
        || volume.devicePointer() == nullptr)
    {
        return false;
    }

    if (vectors.packedPointer() == nullptr)
    {
        return false;
    }

    bool useConstantVectors = false;
    if (chunk.angleCount <= 64)
    {
        useConstantVectors = uploadBpPackedVectorsToConstantAsync(vectors.packedPointer(), chunk.angleCount, stream);
    }

    launchBp3dOptimizedKernel(chunk.deviceData, vectors.packedPointer(), useConstantVectors, chunk.angleCount,
        volumeWidth_, chunk.zOffset, chunk.chunkHeight, volume.devicePointer(), stream);

    return cudaPeekAtLastError() == cudaSuccess;
}

bool Bp3dCudaBackend::backProjectSlab(const DeviceChunkView& chunk, const GpuVectorBuffer& vectors,
    GpuVolumeBuffer& slabVolume, cudaStream_t stream) const
{
    if (chunk.deviceData == nullptr || chunk.chunkHeight <= 0 || chunk.angleCount <= 0 || chunk.detWidth <= 0
        || slabVolume.devicePointer() == nullptr || slabVolume.depth() < chunk.chunkHeight)
    {
        return false;
    }

    if (vectors.packedPointer() == nullptr)
    {
        return false;
    }

    bool useConstantVectors = false;
    if (chunk.angleCount <= 64)
    {
        useConstantVectors = uploadBpPackedVectorsToConstantAsync(vectors.packedPointer(), chunk.angleCount, stream);
    }

    launchBp3dSlabOptimizedKernel(chunk.deviceData, vectors.packedPointer(), useConstantVectors, chunk.angleCount,
        volumeWidth_, chunk.zOffset, chunk.chunkHeight, slabVolume.devicePointer(), stream);

    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace ct
