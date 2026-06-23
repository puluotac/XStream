#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace ct
{

class IReconstructionVolume
{
public:
    virtual ~IReconstructionVolume() = default;

    virtual int depth() const = 0;
    virtual int height() const = 0;
    virtual int width() const = 0;
    virtual size_t voxelCount() const = 0;

    virtual cv::Mat downloadSlice(int zIndex) const = 0;
    virtual bool downloadRange(int zStart, int zCount, std::vector<float>& out) const = 0;
};

} // namespace ct
