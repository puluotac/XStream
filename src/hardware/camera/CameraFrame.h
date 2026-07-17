#pragma once

#include "core/acq/FrameBufferPool.h"

#include <QDateTime>
#include <QString>

#include <opencv2/core.hpp>

namespace ct
{

struct CameraFrame
{
    qint64 frameIndex = 0;
    qint64 logicalIndex = -1;
    qint64 deviceFrameNumber = 0;
    qint64 bofTimestampNs = 0;
    qint64 eofTimestampNs = 0;
    double elapsedMs = 0.0;
    qint64 timestampMs = 0;
    int width = 0;
    int height = 0;
    int bitDepth = 0;
    int bytesPerPixel = 0;
    int strideBytes = 0;
    int cvType = 0;
    QString timingSource;
    FrameBufferPool::LeasePtr bufferLease;
    cv::Mat image;
};

} // namespace ct
