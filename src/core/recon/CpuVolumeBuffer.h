#pragma once

#include "core/recon/IReconstructionVolume.h"

#include <cstddef>
#include <vector>

namespace ct
{

class CpuVolumeBuffer : public IReconstructionVolume
{
public:
    CpuVolumeBuffer() = default;
    ~CpuVolumeBuffer();

    CpuVolumeBuffer(const CpuVolumeBuffer&) = delete;
    CpuVolumeBuffer& operator=(const CpuVolumeBuffer&) = delete;

    bool initialize(int depth, int height, int width);
    void release();
    void clear();

    int depth() const override;
    int height() const override;
    int width() const override;
    size_t voxelCount() const override;

    cv::Mat downloadSlice(int zIndex) const override;
    bool downloadRange(int zStart, int zCount, std::vector<float>& out) const override;
    bool writeSlab(int zStart, int zCount, const std::vector<float>& slab);
    bool copyRangeTo(int zStart, int zCount, float* out) const;
    bool writeSlabFrom(int zStart, int zCount, const float* slab);
    float* rangePointer(int zStart, int zCount);
    const float* rangePointer(int zStart, int zCount) const;

private:
    int depth_ = 0;
    int height_ = 0;
    int width_ = 0;
    size_t voxelCount_ = 0;
    float* data_ = nullptr;
};

} // namespace ct
