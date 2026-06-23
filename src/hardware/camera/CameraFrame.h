#pragma once

#include <QDateTime>

#include <opencv2/core.hpp>

namespace ct
{

struct CameraFrame
{
    qint64 frameIndex = 0;
    qint64 timestampMs = 0;
    int width = 0;
    int height = 0;
    int bitDepth = 0;
    int bytesPerPixel = 0;
    int strideBytes = 0;
    int cvType = 0;
    cv::Mat image;
};

} // namespace ct
