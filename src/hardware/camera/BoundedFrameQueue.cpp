#include "hardware/camera/BoundedFrameQueue.h"

#include <QMutexLocker>

#include <algorithm>

namespace ct
{

BoundedFrameQueue::BoundedFrameQueue(int capacity)
    : capacity_(std::max(1, capacity))
{
}

void BoundedFrameQueue::setCapacity(int capacity)
{
    QMutexLocker lock(&mutex_);
    capacity_ = std::max(1, capacity);
    frames_.clear();
    consumedCount_ = 0;
}

int BoundedFrameQueue::capacity() const
{
    QMutexLocker lock(&mutex_);
    return capacity_;
}

int BoundedFrameQueue::size() const
{
    QMutexLocker lock(&mutex_);
    return static_cast<int>(frames_.size());
}

qint64 BoundedFrameQueue::consumedCount() const
{
    QMutexLocker lock(&mutex_);
    return consumedCount_;
}

void BoundedFrameQueue::clear()
{
    QMutexLocker lock(&mutex_);
    frames_.clear();
    consumedCount_ = 0;
}

bool BoundedFrameQueue::push(CameraFrame frame)
{
    QMutexLocker lock(&mutex_);
    if (static_cast<int>(frames_.size()) >= capacity_)
    {
        return false;
    }
    frames_.push_back(std::move(frame));
    return true;
}

bool BoundedFrameQueue::pop(CameraFrame* frame)
{
    QMutexLocker lock(&mutex_);
    if (frames_.empty())
    {
        return false;
    }
    if (frame != nullptr)
    {
        *frame = std::move(frames_.front());
    }
    frames_.pop_front();
    ++consumedCount_;
    return true;
}

} // namespace ct
