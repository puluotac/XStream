#pragma once

#include "core/config/AppConfig.h"

#include <opencv2/core.hpp>

#include <QString>
#include <vector>

namespace ct
{

struct ProjectionFrame
{
    int logicalIndex = -1;
    int fileIndex = -1;
    QString path;
    cv::Mat image;
};

struct CorrectionContext
{
    CorrectionMode mode = CorrectionMode::FlatDark;
    cv::Mat flat;
    cv::Mat dark;
    int cropMargin = 50;
};

struct ChunkSinogram
{
    std::vector<float> hostData;
    int chunkHeight = 0;
    int angleCount = 0;
    int detWidth = 0;
    int zOffset = 0;
};

struct DeviceChunkView
{
    const float* deviceData = nullptr;
    int chunkHeight = 0;
    int angleCount = 0;
    int detWidth = 0;
    int zOffset = 0;
};

} // namespace ct
