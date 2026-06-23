#pragma once

#include "core/recon/IReconstructionVolume.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

#include <cuda_runtime_api.h>

namespace ct
{

class GpuVolumeBuffer : public IReconstructionVolume
{
public:
    GpuVolumeBuffer() = default;
    ~GpuVolumeBuffer();

    GpuVolumeBuffer(const GpuVolumeBuffer&) = delete;
    GpuVolumeBuffer& operator=(const GpuVolumeBuffer&) = delete;

    bool initialize(int depth, int height, int width);
    void clear();

    int depth() const override;
    int height() const override;
    int width() const override;
    size_t voxelCount() const override;

    float* devicePointer();
    const float* devicePointer() const;

    bool uploadAll(const std::vector<float>& input);
    bool uploadAllFromHost(const float* input);
    bool uploadRange(int zStart, int zCount, const std::vector<float>& input);
    bool uploadRangeAsync(int zStart, int zCount, const float* input, cudaStream_t stream);
    cv::Mat downloadSlice(int zIndex) const override;
    bool downloadRange(int zStart, int zCount, std::vector<float>& out) const override;
    bool downloadAllToHost(float* out) const;
    bool downloadAll(std::vector<float>& out) const;
    bool downloadRangeAsync(int zStart, int zCount, float* out, cudaStream_t stream) const;

private:
    int depth_ = 0;
    int height_ = 0;
    int width_ = 0;
    float* devicePtr_ = nullptr;
};

} // namespace ct
