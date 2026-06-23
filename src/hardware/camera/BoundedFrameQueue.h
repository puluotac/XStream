#pragma once

#include "hardware/camera/CameraFrame.h"

#include <QMutex>

#include <deque>

namespace ct
{

class BoundedFrameQueue
{
public:
    explicit BoundedFrameQueue(int capacity = 256);

    void setCapacity(int capacity);
    int capacity() const;
    int size() const;
    qint64 consumedCount() const;
    void clear();
    bool push(CameraFrame frame);
    bool pop(CameraFrame* frame);

private:
    mutable QMutex mutex_;
    std::deque<CameraFrame> frames_;
    int capacity_ = 256;
    qint64 consumedCount_ = 0;
};

} // namespace ct
