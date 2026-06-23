#include "core/recon/CpuVolumeBuffer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ct
{

namespace
{

bool checkedVoxelCount(int depth, int height, int width, size_t* outCount)
{
    if (depth <= 0 || height <= 0 || width <= 0 || outCount == nullptr)
    {
        return false;
    }

    const size_t d = static_cast<size_t>(depth);
    const size_t h = static_cast<size_t>(height);
    const size_t w = static_cast<size_t>(width);
    if (d > std::numeric_limits<size_t>::max() / h || d * h > std::numeric_limits<size_t>::max() / w)
    {
        return false;
    }

    const size_t voxels = d * h * w;
    if (voxels > std::numeric_limits<size_t>::max() / sizeof(float))
    {
        return false;
    }

    *outCount = voxels;
    return true;
}

float* allocateFloatVolume(size_t voxelCount)
{
    if (voxelCount == 0)
    {
        return nullptr;
    }

    const size_t bytes = voxelCount * sizeof(float);
#ifdef _WIN32
    return static_cast<float*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, bytes) != 0)
    {
        return nullptr;
    }
    std::memset(ptr, 0, bytes);
    return static_cast<float*>(ptr);
#endif
}

void freeFloatVolume(float* data)
{
    if (data == nullptr)
    {
        return;
    }

#ifdef _WIN32
    VirtualFree(data, 0, MEM_RELEASE);
#else
    std::free(data);
#endif
}

} // namespace

CpuVolumeBuffer::~CpuVolumeBuffer()
{
    release();
}

bool CpuVolumeBuffer::initialize(int depth, int height, int width)
{
    release();

    size_t voxels = 0;
    if (!checkedVoxelCount(depth, height, width, &voxels))
    {
        return false;
    }

    float* volume = allocateFloatVolume(voxels);
    if (volume == nullptr)
    {
        return false;
    }

    depth_ = depth;
    height_ = height;
    width_ = width;
    voxelCount_ = voxels;
    data_ = volume;
    return true;
}

void CpuVolumeBuffer::release()
{
    freeFloatVolume(data_);
    data_ = nullptr;
    voxelCount_ = 0;
    depth_ = 0;
    height_ = 0;
    width_ = 0;
}

void CpuVolumeBuffer::clear()
{
    if (data_ != nullptr)
    {
        std::memset(data_, 0, voxelCount_ * sizeof(float));
    }
}

int CpuVolumeBuffer::depth() const
{
    return depth_;
}

int CpuVolumeBuffer::height() const
{
    return height_;
}

int CpuVolumeBuffer::width() const
{
    return width_;
}

size_t CpuVolumeBuffer::voxelCount() const
{
    return voxelCount_;
}

cv::Mat CpuVolumeBuffer::downloadSlice(int zIndex) const
{
    cv::Mat slice(height_, width_, CV_32F, cv::Scalar(0.0f));
    if (data_ == nullptr || zIndex < 0 || zIndex >= depth_)
    {
        return slice;
    }

    const size_t offset = static_cast<size_t>(zIndex) * height_ * width_;
    std::memcpy(slice.ptr<float>(), data_ + offset, static_cast<size_t>(height_) * width_ * sizeof(float));
    return slice;
}

bool CpuVolumeBuffer::downloadRange(int zStart, int zCount, std::vector<float>& out) const
{
    if (data_ == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return false;
    }

    const size_t elements = static_cast<size_t>(zCount) * height_ * width_;
    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    out.assign(data_ + offset, data_ + offset + elements);
    return true;
}

bool CpuVolumeBuffer::writeSlab(int zStart, int zCount, const std::vector<float>& slab)
{
    if (data_ == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return false;
    }

    const size_t elements = static_cast<size_t>(zCount) * height_ * width_;
    if (slab.size() < elements)
    {
        return false;
    }

    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    std::memcpy(data_ + offset, slab.data(), elements * sizeof(float));
    return true;
}

bool CpuVolumeBuffer::copyRangeTo(int zStart, int zCount, float* out) const
{
    if (data_ == nullptr || out == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return false;
    }

    const size_t elements = static_cast<size_t>(zCount) * height_ * width_;
    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    std::memcpy(out, data_ + offset, elements * sizeof(float));
    return true;
}

bool CpuVolumeBuffer::writeSlabFrom(int zStart, int zCount, const float* slab)
{
    if (data_ == nullptr || slab == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return false;
    }

    const size_t elements = static_cast<size_t>(zCount) * height_ * width_;
    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    std::memcpy(data_ + offset, slab, elements * sizeof(float));
    return true;
}

float* CpuVolumeBuffer::rangePointer(int zStart, int zCount)
{
    if (data_ == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return nullptr;
    }

    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    return data_ + offset;
}

const float* CpuVolumeBuffer::rangePointer(int zStart, int zCount) const
{
    if (data_ == nullptr || zStart < 0 || zCount <= 0 || (zStart + zCount) > depth_)
    {
        return nullptr;
    }

    const size_t offset = static_cast<size_t>(zStart) * height_ * width_;
    return data_ + offset;
}

} // namespace ct
