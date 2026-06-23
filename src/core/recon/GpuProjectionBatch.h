#pragma once

#include "core/common/ProjectionTypes.h"

#include <opencv2/core.hpp>

#include <cstddef>

#include <cuda_runtime_api.h>

namespace ct
{

class HostPinnedProjectionBatch
{
public:
    HostPinnedProjectionBatch() = default;
    ~HostPinnedProjectionBatch();

    HostPinnedProjectionBatch(const HostPinnedProjectionBatch&) = delete;
    HostPinnedProjectionBatch& operator=(const HostPinnedProjectionBatch&) = delete;

    bool initialize(int angleCapacity, int height, int width);
    bool copyFrame(int slot, const cv::Mat& image);
    void clear();

    float* rawAngleMajorHost();
    const float* rawAngleMajorHost() const;

    int angleCapacity() const;
    int height() const;
    int width() const;
    size_t rawFloatCount(int angleCount) const;

private:
    int angleCapacity_ = 0;
    int height_ = 0;
    int width_ = 0;
    float* hostRawAngleMajor_ = nullptr;
};

class GpuProjectionBatch
{
public:
    GpuProjectionBatch() = default;
    ~GpuProjectionBatch();

    GpuProjectionBatch(const GpuProjectionBatch&) = delete;
    GpuProjectionBatch& operator=(const GpuProjectionBatch&) = delete;

    bool initialize(int angleCapacity, int height, int width);
    void clear();

    bool uploadRawFromHostAsync(const HostPinnedProjectionBatch& hostBatch, int angleCount, cudaStream_t stream);

    float* rawAngleMajor();
    const float* rawAngleMajor() const;
    float* reconLayout();
    const float* reconLayout() const;

    int angleCapacity() const;
    int currentAngleCount() const;
    int height() const;
    int width() const;

    size_t rawBytes(int angleCount) const;
    size_t reconBytes(int angleCount) const;

    DeviceChunkView makeChunkView(int zStart, int chunkHeight) const;

private:
    int angleCapacity_ = 0;
    int currentAngleCount_ = 0;
    int height_ = 0;
    int width_ = 0;
    float* deviceRawAngleMajor_ = nullptr;
    float* deviceReconLayout_ = nullptr;
};

} // namespace ct
