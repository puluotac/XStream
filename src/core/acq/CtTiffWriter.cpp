#include "core/acq/CtTiffWriter.h"

#include "core/io/ImageFileIo.h"

#include <QDateTime>

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ct
{

CtTiffWriter::CtTiffWriter(QObject* parent)
    : QObject(parent)
{
}

CtTiffWriter::~CtTiffWriter()
{
    stop();
}

void CtTiffWriter::configure(int threadCount, int maxQueuedFrames)
{
    threadCount_ = std::clamp(threadCount, 1, 64);
    maxQueuedFrames_ = std::max(1, maxQueuedFrames);
}

void CtTiffWriter::start()
{
    stop();
    {
        QMutexLocker lock(&mutex_);
        queue_.clear();
        savedCount_ = 0;
        accepting_ = true;
        finishRequested_ = false;
        stopRequested_ = false;
        failed_ = false;
        errorMessage_.clear();
    }
    lastQueueSignalMs_.store(0, std::memory_order_release);

    workers_.reserve(static_cast<size_t>(threadCount_));
    for (int i = 0; i < threadCount_; ++i)
    {
        workers_.emplace_back([this, i]() { workerLoop(i); });
    }
    emit logMessage(QStringLiteral("CT TIFF writer started with %1 threads, queue capacity %2 frames.")
        .arg(threadCount_)
        .arg(maxQueuedFrames_));
    emitQueueChangedThrottled(true);
}

void CtTiffWriter::enqueue(const CtFrameSaveItem& item)
{
    {
        QMutexLocker lock(&mutex_);
        while (!stopRequested_ && !failed_ && accepting_ && queue_.size() >= maxQueuedFrames_)
        {
            emit logMessage(QStringLiteral("CT save queue is full (%1 frames); acquisition waits.").arg(maxQueuedFrames_));
            notFull_.wait(&mutex_, 250);
        }
        if (stopRequested_ || failed_ || !accepting_)
        {
            return;
        }
        queue_.enqueue(item);
        notEmpty_.wakeOne();
    }
    emitQueueChangedThrottled();
}

bool CtTiffWriter::tryEnqueue(const CtFrameSaveItem& item, QString* errorMessage)
{
    {
        QMutexLocker lock(&mutex_);
        if (stopRequested_ || failed_ || !accepting_)
        {
            if (errorMessage)
            {
                *errorMessage = failed_ && !errorMessage_.isEmpty()
                    ? errorMessage_
                    : QStringLiteral("CT TIFF writer is not accepting frames.");
            }
            return false;
        }
        if (queue_.size() >= maxQueuedFrames_)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("CT TIFF writer queue is full at %1 frames.").arg(maxQueuedFrames_);
            }
            return false;
        }
        queue_.enqueue(item);
        notEmpty_.wakeOne();
    }
    emitQueueChangedThrottled();
    return true;
}

void CtTiffWriter::emitQueueChangedThrottled(bool force)
{
    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    qint64 observedMs = lastQueueSignalMs_.load(std::memory_order_acquire);
    while (!force && observedMs > 0 && nowMs - observedMs < 100)
    {
        return;
    }
    while (!lastQueueSignalMs_.compare_exchange_weak(observedMs, nowMs,
        std::memory_order_acq_rel, std::memory_order_acquire))
    {
        if (!force && observedMs > 0 && nowMs - observedMs < 100)
        {
            return;
        }
    }
    int queued = 0;
    int saved = 0;
    {
        QMutexLocker lock(&mutex_);
        queued = queue_.size();
        saved = savedCount_;
    }
    emit queueChanged(queued, saved);
}

void CtTiffWriter::finish()
{
    {
        QMutexLocker lock(&mutex_);
        accepting_ = false;
        finishRequested_ = true;
        notEmpty_.wakeAll();
        notFull_.wakeAll();
    }

    for (std::thread& worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    workers_.clear();

    bool ok = false;
    QString message;
    {
        QMutexLocker lock(&mutex_);
        ok = !failed_ && !stopRequested_;
        message = ok
            ? QStringLiteral("CT TIFF writer finished: %1 frames saved.").arg(savedCount_)
            : QStringLiteral("CT TIFF writer failed/stopped: %1").arg(errorMessage_);
    }
    emitQueueChangedThrottled(true);
    emit finished(ok, message);
}

void CtTiffWriter::stop()
{
    {
        QMutexLocker lock(&mutex_);
        accepting_ = false;
        stopRequested_ = true;
        notEmpty_.wakeAll();
        notFull_.wakeAll();
    }
    for (std::thread& worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    workers_.clear();
}

void CtTiffWriter::workerLoop(int workerIndex)
{
    Q_UNUSED(workerIndex);

    for (;;)
    {
        CtFrameSaveItem item;
        {
            QMutexLocker lock(&mutex_);
            while (!stopRequested_ && !failed_ && queue_.isEmpty() && !finishRequested_)
            {
                notEmpty_.wait(&mutex_);
            }
            if (stopRequested_ || failed_)
            {
                return;
            }
            if (queue_.isEmpty() && finishRequested_)
            {
                return;
            }
            item = std::move(queue_.dequeue());
            notFull_.wakeOne();
        }
        emitQueueChangedThrottled();

        try
        {
            cv::Mat image;
            if (item.bufferLease && item.bufferLease->data() != nullptr
                && item.width > 0 && item.height > 0 && item.strideBytes > 0)
            {
                image = cv::Mat(item.height, item.width, item.cvType,
                    item.bufferLease->data(), static_cast<size_t>(item.strideBytes));
            }
            else if (item.ownedImage)
            {
                image = *item.ownedImage;
            }
            if (image.empty())
            {
                throw std::runtime_error("CT writer item has no image buffer.");
            }
            QString error;
            if (!writeImageFile(item.filePath, image, &error))
            {
                throw std::runtime_error(error.toLocal8Bit().constData());
            }
        }
        catch (const std::exception& ex)
        {
            QMutexLocker lock(&mutex_);
            failed_ = true;
            errorMessage_ = QStringLiteral("%1: %2").arg(item.filePath, QString::fromLocal8Bit(ex.what()));
            notEmpty_.wakeAll();
            notFull_.wakeAll();
            return;
        }

        {
            QMutexLocker lock(&mutex_);
            ++savedCount_;
        }
        emitQueueChangedThrottled();
    }
}

} // namespace ct
