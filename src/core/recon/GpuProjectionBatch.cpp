#include "core/recon/GpuProjectionBatch.h"

#include <cuda_runtime.h>

#include <algorithm>

namespace ct
{

HostPinnedProjectionBatch::~HostPinnedProjectionBatch()
{
    if (hostRawAngleMajor_ != nullptr)
    {
        cudaFreeHost(hostRawAngleMajor_);
        hostRawAngleMajor_ = nullptr;
    }
}

bool HostPinnedProjectionBatch::initialize(int angleCapacity, int height, int width)
{
    if (hostRawAngleMajor_ != nullptr)
    {
        cudaFreeHost(hostRawAngleMajor_);
        hostRawAngleMajor_ = nullptr;
    }

    angleCapacity_ = angleCapacity;
    height_ = height;
    width_ = width;

    const size_t floatCount = rawFloatCount(angleCapacity_);
    if (floatCount == 0)
    {
        return false;
    }

    return cudaMallocHost(&hostRawAngleMajor_, floatCount * sizeof(float)) == cudaSuccess;
}

bool HostPinnedProjectionBatch::copyFrame(int slot, const cv::Mat& image)
{
    if (hostRawAngleMajor_ == nullptr || slot < 0 || slot >= angleCapacity_ || image.empty())
    {
        return false;
    }

    if (image.rows != height_ || image.cols != width_ || image.channels() != 1)
    {
        return false;
    }

    cv::Mat destination(height_, width_, CV_32F,
        hostRawAngleMajor_ + static_cast<size_t>(slot) * height_ * width_);
    image.convertTo(destination, CV_32F);
    return true;
}

void HostPinnedProjectionBatch::clear()
{
    if (hostRawAngleMajor_ != nullptr)
    {
        std::fill(hostRawAngleMajor_, hostRawAngleMajor_ + rawFloatCount(angleCapacity_), 0.0f);
    }
}

float* HostPinnedProjectionBatch::rawAngleMajorHost()
{
    return hostRawAngleMajor_;
}

const float* HostPinnedProjectionBatch::rawAngleMajorHost() const
{
    return hostRawAngleMajor_;
}

int HostPinnedProjectionBatch::angleCapacity() const
{
    return angleCapacity_;
}

int HostPinnedProjectionBatch::height() const
{
    return height_;
}

int HostPinnedProjectionBatch::width() const
{
    return width_;
}

size_t HostPinnedProjectionBatch::rawFloatCount(int angleCount) const
{
    return static_cast<size_t>(angleCount) * height_ * width_;
}

GpuProjectionBatch::~GpuProjectionBatch()
{
    if (deviceRawAngleMajor_ != nullptr)
    {
        cudaFree(deviceRawAngleMajor_);
        deviceRawAngleMajor_ = nullptr;
    }
    if (deviceReconLayout_ != nullptr)
    {
        cudaFree(deviceReconLayout_);
        deviceReconLayout_ = nullptr;
    }
}

bool GpuProjectionBatch::initialize(int angleCapacity, int height, int width)
{
    if (deviceRawAngleMajor_ != nullptr)
    {
        cudaFree(deviceRawAngleMajor_);
        deviceRawAngleMajor_ = nullptr;
    }
    if (deviceReconLayout_ != nullptr)
    {
        cudaFree(deviceReconLayout_);
        deviceReconLayout_ = nullptr;
    }

    angleCapacity_ = angleCapacity;
    currentAngleCount_ = 0;
    height_ = height;
    width_ = width;

    const size_t rawByteCount = rawBytes(angleCapacity_);
    const size_t reconByteCount = reconBytes(angleCapacity_);
    if (rawByteCount == 0 || reconByteCount == 0)
    {
        return false;
    }

    if (cudaMalloc(&deviceRawAngleMajor_, rawByteCount) != cudaSuccess)
    {
        deviceRawAngleMajor_ = nullptr;
        return false;
    }

    if (cudaMalloc(&deviceReconLayout_, reconByteCount) != cudaSuccess)
    {
        cudaFree(deviceRawAngleMajor_);
        deviceRawAngleMajor_ = nullptr;
        deviceReconLayout_ = nullptr;
        return false;
    }

    clear();
    return true;
}

void GpuProjectionBatch::clear()
{
    if (deviceRawAngleMajor_ != nullptr)
    {
        cudaMemset(deviceRawAngleMajor_, 0, rawBytes(angleCapacity_));
    }
    if (deviceReconLayout_ != nullptr)
    {
        cudaMemset(deviceReconLayout_, 0, reconBytes(angleCapacity_));
    }
    currentAngleCount_ = 0;
}

bool GpuProjectionBatch::uploadRawFromHostAsync(const HostPinnedProjectionBatch& hostBatch, int angleCount, cudaStream_t stream)
{
    if (deviceRawAngleMajor_ == nullptr || hostBatch.rawAngleMajorHost() == nullptr)
    {
        return false;
    }

    if (angleCount <= 0 || angleCount > angleCapacity_)
    {
        return false;
    }

    currentAngleCount_ = angleCount;
    return cudaMemcpyAsync(deviceRawAngleMajor_, hostBatch.rawAngleMajorHost(),
        rawBytes(angleCount), cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

float* GpuProjectionBatch::rawAngleMajor()
{
    return deviceRawAngleMajor_;
}

const float* GpuProjectionBatch::rawAngleMajor() const
{
    return deviceRawAngleMajor_;
}

float* GpuProjectionBatch::reconLayout()
{
    return deviceReconLayout_;
}

const float* GpuProjectionBatch::reconLayout() const
{
    return deviceReconLayout_;
}

int GpuProjectionBatch::angleCapacity() const
{
    return angleCapacity_;
}

int GpuProjectionBatch::currentAngleCount() const
{
    return currentAngleCount_;
}

int GpuProjectionBatch::height() const
{
    return height_;
}

int GpuProjectionBatch::width() const
{
    return width_;
}

size_t GpuProjectionBatch::rawBytes(int angleCount) const
{
    return static_cast<size_t>(angleCount) * height_ * width_ * sizeof(float);
}

size_t GpuProjectionBatch::reconBytes(int angleCount) const
{
    return static_cast<size_t>(height_) * angleCount * width_ * sizeof(float);
}

DeviceChunkView GpuProjectionBatch::makeChunkView(int zStart, int chunkHeight) const
{
    if (deviceReconLayout_ == nullptr || currentAngleCount_ <= 0 || zStart < 0 || chunkHeight <= 0
        || (zStart + chunkHeight) > height_)
    {
        return {};
    }

    DeviceChunkView view;
    view.deviceData = deviceReconLayout_ + static_cast<size_t>(zStart) * currentAngleCount_ * width_;
    view.chunkHeight = chunkHeight;
    view.angleCount = currentAngleCount_;
    view.detWidth = width_;
    view.zOffset = zStart;
    return view;
}

} // namespace ct
