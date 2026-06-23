#include "core/recon/GpuVectorBuffer.h"

#include <cuda_runtime.h>

namespace ct
{

GpuVectorBuffer::~GpuVectorBuffer()
{
    if (devicePackedVectors_ != nullptr)
    {
        cudaFree(devicePackedVectors_);
        devicePackedVectors_ = nullptr;
    }
}

bool GpuVectorBuffer::initialize(int maxAngles)
{
    if (devicePackedVectors_ != nullptr)
    {
        cudaFree(devicePackedVectors_);
        devicePackedVectors_ = nullptr;
    }

    maxAngles_ = maxAngles;
    currentAngleCount_ = 0;
    const size_t packedCount = static_cast<size_t>(maxAngles_);
    if (packedCount == 0)
    {
        return false;
    }

    hostPackedVectors_.assign(packedCount, make_float4(0.0f, 0.0f, 0.0f, 0.0f));

    if (cudaMalloc(&devicePackedVectors_, packedCount * sizeof(float4)) != cudaSuccess)
    {
        devicePackedVectors_ = nullptr;
        return false;
    }

    return true;
}

bool GpuVectorBuffer::uploadAsync(const std::vector<double>& hostVectors, cudaStream_t stream)
{
    if (devicePackedVectors_ == nullptr || hostVectors.empty() || (hostVectors.size() % 12) != 0)
    {
        return false;
    }

    currentAngleCount_ = static_cast<int>(hostVectors.size() / 12);
    if (currentAngleCount_ > maxAngles_)
    {
        return false;
    }

    for (int angle = 0; angle < currentAngleCount_; ++angle)
    {
        const size_t base = static_cast<size_t>(angle) * 12;
        hostPackedVectors_[static_cast<size_t>(angle)] = make_float4(
            static_cast<float>(hostVectors[base + 3]),
            static_cast<float>(hostVectors[base + 4]),
            static_cast<float>(hostVectors[base + 6]),
            static_cast<float>(hostVectors[base + 7]));
    }

    return cudaMemcpyAsync(devicePackedVectors_, hostPackedVectors_.data(),
        static_cast<size_t>(currentAngleCount_) * sizeof(float4),
        cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

const float4* GpuVectorBuffer::packedPointer() const
{
    return devicePackedVectors_;
}

int GpuVectorBuffer::currentAngleCount() const
{
    return currentAngleCount_;
}

} // namespace ct
