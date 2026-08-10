#pragma once

#include <QString>

#include <QtGlobal>

#include <memory>

namespace ct
{

class FrameBufferPoolState;

class FrameBufferLease
{
public:
    ~FrameBufferLease();

    FrameBufferLease(const FrameBufferLease&) = delete;
    FrameBufferLease& operator=(const FrameBufferLease&) = delete;

    unsigned char* data() const;
    qsizetype size() const;
    int slotIndex() const;

private:
    friend class FrameBufferPool;
    FrameBufferLease(std::shared_ptr<FrameBufferPoolState> state, int slotIndex);

    std::shared_ptr<FrameBufferPoolState> state_;
    int slotIndex_ = -1;
};

struct FrameBufferPoolStats
{
    int capacity = 0;
    int available = 0;
    int inUse = 0;
    int peakInUse = 0;
    qsizetype frameBytes = 0;
    quint64 totalBytes = 0;
};

class FrameBufferPool
{
public:
    using LeasePtr = std::shared_ptr<FrameBufferLease>;

    bool configure(qsizetype frameBytes, int capacity, bool preTouch, QString* errorMessage = nullptr);
    LeasePtr tryAcquire();
    FrameBufferPoolStats stats() const;
    void resetPeak();
    void clear();

private:
    std::shared_ptr<FrameBufferPoolState> state_;
};

} // namespace ct
