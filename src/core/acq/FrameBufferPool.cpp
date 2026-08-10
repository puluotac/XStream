#include "core/acq/FrameBufferPool.h"

#include <QMutex>
#include <QMutexLocker>
#include <QVector>

#include <algorithm>
#include <limits>
#include <new>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ct
{

class FrameBufferPoolState
{
public:
    ~FrameBufferPoolState()
    {
#ifdef Q_OS_WIN
        if (memory != nullptr)
        {
            VirtualFree(memory, 0, MEM_RELEASE);
        }
#else
        ::operator delete(memory);
#endif
    }

    void release(int slotIndex)
    {
        QMutexLocker lock(&mutex);
        if (slotIndex < 0 || slotIndex >= capacity)
        {
            return;
        }
        freeSlots.append(slotIndex);
        inUse = std::max(0, inUse - 1);
    }

    mutable QMutex mutex;
    unsigned char* memory = nullptr;
    qsizetype frameBytes = 0;
    int capacity = 0;
    int inUse = 0;
    int peakInUse = 0;
    quint64 totalBytes = 0;
    QVector<int> freeSlots;
};

FrameBufferLease::FrameBufferLease(std::shared_ptr<FrameBufferPoolState> state, int slotIndex)
    : state_(std::move(state))
    , slotIndex_(slotIndex)
{
}

FrameBufferLease::~FrameBufferLease()
{
    if (state_)
    {
        state_->release(slotIndex_);
    }
}

unsigned char* FrameBufferLease::data() const
{
    if (!state_ || slotIndex_ < 0)
    {
        return nullptr;
    }
    return state_->memory + static_cast<qsizetype>(slotIndex_) * state_->frameBytes;
}

qsizetype FrameBufferLease::size() const
{
    return state_ ? state_->frameBytes : 0;
}

int FrameBufferLease::slotIndex() const
{
    return slotIndex_;
}

bool FrameBufferPool::configure(qsizetype frameBytes, int capacity, bool preTouch, QString* errorMessage)
{
    if (frameBytes <= 0 || capacity <= 0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Frame pool size and capacity must be positive.");
        }
        return false;
    }

    const quint64 totalBytes = static_cast<quint64>(frameBytes) * static_cast<quint64>(capacity);
    if (totalBytes > static_cast<quint64>(std::numeric_limits<size_t>::max()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Frame pool allocation size exceeds the process address space.");
        }
        return false;
    }

    if (state_)
    {
        const FrameBufferPoolStats current = stats();
        if (current.inUse > 0)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("Cannot resize the frame pool while %1 slot(s) are in use.")
                    .arg(current.inUse);
            }
            return false;
        }
    }

    auto next = std::make_shared<FrameBufferPoolState>();
#ifdef Q_OS_WIN
    next->memory = static_cast<unsigned char*>(VirtualAlloc(
        nullptr,
        static_cast<SIZE_T>(totalBytes),
        MEM_RESERVE | MEM_COMMIT,
        PAGE_READWRITE));
#else
    next->memory = static_cast<unsigned char*>(::operator new(static_cast<size_t>(totalBytes), std::nothrow));
#endif
    if (next->memory == nullptr)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Failed to allocate %1 bytes for the frame pool.").arg(totalBytes);
        }
        return false;
    }

    next->frameBytes = frameBytes;
    next->capacity = capacity;
    next->totalBytes = totalBytes;
    next->freeSlots.reserve(capacity);
    for (int slot = capacity - 1; slot >= 0; --slot)
    {
        next->freeSlots.append(slot);
    }

    if (preTouch)
    {
        constexpr qsizetype pageStride = 4096;
        for (quint64 offset = 0; offset < totalBytes; offset += pageStride)
        {
            next->memory[offset] = 0;
        }
        next->memory[totalBytes - 1] = 0;
    }

    state_ = std::move(next);
    return true;
}

FrameBufferPool::LeasePtr FrameBufferPool::tryAcquire()
{
    const std::shared_ptr<FrameBufferPoolState> state = state_;
    if (!state)
    {
        return {};
    }

    int slotIndex = -1;
    {
        QMutexLocker lock(&state->mutex);
        if (state->freeSlots.isEmpty())
        {
            return {};
        }
        slotIndex = state->freeSlots.takeLast();
        ++state->inUse;
        state->peakInUse = std::max(state->peakInUse, state->inUse);
    }

    try
    {
        return std::shared_ptr<FrameBufferLease>(new FrameBufferLease(state, slotIndex));
    }
    catch (...)
    {
        state->release(slotIndex);
        throw;
    }
}

FrameBufferPoolStats FrameBufferPool::stats() const
{
    FrameBufferPoolStats result;
    const std::shared_ptr<FrameBufferPoolState> state = state_;
    if (!state)
    {
        return result;
    }
    QMutexLocker lock(&state->mutex);
    result.capacity = state->capacity;
    result.available = state->freeSlots.size();
    result.inUse = state->inUse;
    result.peakInUse = state->peakInUse;
    result.frameBytes = state->frameBytes;
    result.totalBytes = state->totalBytes;
    return result;
}

void FrameBufferPool::resetPeak()
{
    const std::shared_ptr<FrameBufferPoolState> state = state_;
    if (!state)
    {
        return;
    }
    QMutexLocker lock(&state->mutex);
    state->peakInUse = state->inUse;
}

void FrameBufferPool::clear()
{
    if (!state_ || stats().inUse == 0)
    {
        state_.reset();
    }
}

} // namespace ct
