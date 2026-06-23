#include "core/recon/GpuVolumeBuffer.h"

#include <cuda_runtime.h>

namespace ct
{

GpuVolumeBuffer::~GpuVolumeBuffer()
{
    if (devicePtr_ != nullptr)
    {
        cudaFree(devicePtr_);
        devicePtr_ = nullptr;
    }
}

bool GpuVolumeBuffer::initialize(int depth, int height, int width)
{
    depth_ = depth;
    height_ = height;
    width_ = width;

    const size_t bytes = voxelCount() * sizeof(float);
    if (cudaMalloc(&devicePtr_, bytes) != cudaSuccess)
    {
        devicePtr_ = nullptr;
        return false;
    }

    clear();
    return true;
}

void GpuVolumeBuffer::clear()
{
    if (devicePtr_ != nullptr)
    {
        cudaMemset(devicePtr_, 0, voxelCount() * sizeof(float));
    }
}

int GpuVolumeBuffer::depth() const
{
    return depth_;
}

int GpuVolumeBuffer::height() const
{
    return height_;
}

int GpuVolumeBuffer::width() const
{
    return width_;
}

size_t GpuVolumeBuffer::voxelCount() const
{
    return static_cast<size_t>(depth_) * height_ * width_;
}

float* GpuVolumeBuffer::devicePointer()
{
    return devicePtr_;
}

const float* GpuVolumeBuffer::devicePointer() const
{
    return devicePtr_;
}

bool GpuVolumeBuffer::uploadAll(const std::vector<float>& input)
{
    if (devicePtr_ == nullptr || input.size() < voxelCount())
    {
        return false;
    }

    return cudaMemcpy(devicePtr_, input.data(), voxelCount() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool GpuVolumeBuffer::uploadAllFromHost(const float* input)
{
    if (devicePtr_ == nullptr || input == nullptr)
    {
        return false;
    }

    return cudaMemcpy(devicePtr_, input, voxelCount() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool GpuVolumeBuffer::uploadRange(int zStart, int zCount, const std::vector<float>& input)
{
    if (devicePtr_ == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return false;
    }

    const size_t elements = static_cast<size_t>(zCount) * height_ * width_;
    if (input.size() < elements)
    {
        return false;
    }

    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    return cudaMemcpy(devicePtr_ + offset, input.data(), elements * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool GpuVolumeBuffer::uploadRangeAsync(int zStart, int zCount, const float* input, cudaStream_t stream)
{
    if (devicePtr_ == nullptr || input == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return false;
    }

    const size_t elements = static_cast<size_t>(zCount) * height_ * width_;
    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    return cudaMemcpyAsync(devicePtr_ + offset, input, elements * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

cv::Mat GpuVolumeBuffer::downloadSlice(int zIndex) const
{
    cv::Mat slice(height_, width_, CV_32F, cv::Scalar(0.0f));
    if (devicePtr_ == nullptr || zIndex < 0 || zIndex >= depth_)
    {
        return slice;
    }

    const size_t offset = static_cast<size_t>(zIndex) * height_ * width_;
    cudaMemcpy(slice.ptr<float>(), devicePtr_ + offset, static_cast<size_t>(height_) * width_ * sizeof(float), cudaMemcpyDeviceToHost);
    return slice;
}

bool GpuVolumeBuffer::downloadRange(int zStart, int zCount, std::vector<float>& out) const
{
    if (devicePtr_ == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return false;
    }

    const size_t elements = static_cast<size_t>(zCount) * height_ * width_;
    out.resize(elements);
    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    return cudaMemcpy(out.data(), devicePtr_ + offset, elements * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool GpuVolumeBuffer::downloadAllToHost(float* out) const
{
    if (devicePtr_ == nullptr || out == nullptr)
    {
        return false;
    }

    return cudaMemcpy(out, devicePtr_, voxelCount() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool GpuVolumeBuffer::downloadAll(std::vector<float>& out) const
{
    if (devicePtr_ == nullptr)
    {
        return false;
    }

    out.resize(voxelCount());
    return cudaMemcpy(out.data(), devicePtr_, out.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool GpuVolumeBuffer::downloadRangeAsync(int zStart, int zCount, float* out, cudaStream_t stream) const
{
    if (devicePtr_ == nullptr || out == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return false;
    }

    const size_t elements = static_cast<size_t>(zCount) * height_ * width_;
    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    return cudaMemcpyAsync(out, devicePtr_ + offset, elements * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

} // namespace ct
