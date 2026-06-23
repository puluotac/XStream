#pragma once

#include "core/common/ProjectionTypes.h"
#include "core/config/AppConfig.h"

#include <cuda_runtime_api.h>

namespace ct
{

class GpuVolumeBuffer;
class GpuVectorBuffer;

class Bp3dCudaBackend
{
public:
    bool initialize(const GeometryConfig& geometry, const ComputeConfig& compute);
    bool backProjectChunk(const DeviceChunkView& chunk, const GpuVectorBuffer& vectors,
        GpuVolumeBuffer& volume, cudaStream_t stream) const;
    bool backProjectSlab(const DeviceChunkView& chunk, const GpuVectorBuffer& vectors,
        GpuVolumeBuffer& slabVolume, cudaStream_t stream) const;

private:
    int volumeWidth_ = 0;
};

} // namespace ct
