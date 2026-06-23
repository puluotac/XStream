#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime_api.h>

namespace ct
{

class GpuVectorBuffer
{
public:
    GpuVectorBuffer() = default;
    ~GpuVectorBuffer();

    GpuVectorBuffer(const GpuVectorBuffer&) = delete;
    GpuVectorBuffer& operator=(const GpuVectorBuffer&) = delete;

    bool initialize(int maxAngles);
    bool uploadAsync(const std::vector<double>& hostVectors, cudaStream_t stream);

    const float4* packedPointer() const;
    int currentAngleCount() const;

private:
    int maxAngles_ = 0;
    int currentAngleCount_ = 0;
    float4* devicePackedVectors_ = nullptr;
    std::vector<float4> hostPackedVectors_;
};

} // namespace ct
