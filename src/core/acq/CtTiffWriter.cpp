#include "core/acq/CtTiffWriter.h"

#include <opencv2/imgcodecs.hpp>

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

    workers_.reserve(static_cast<size_t>(threadCount_));
    for (int i = 0; i < threadCount_; ++i)
    {
        workers_.emplace_back([this, i]() { workerLoop(i); });
    }
    emit logMessage(QStringLiteral("CT TIFF writer started with %1 threads, queue capacity %2 frames.")
        .arg(threadCount_)
        .arg(maxQueuedFrames_));
}

void CtTiffWriter::enqueue(const CtFrameSaveItem& item)
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
    emit queueChanged(queue_.size(), savedCount_);
    notEmpty_.wakeOne();
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
            emit queueChanged(queue_.size(), savedCount_);
        }

        try
        {
            const std::vector<int> params { cv::IMWRITE_TIFF_COMPRESSION, 1 };
            if (item.image.empty() || !cv::imwrite(item.filePath.toStdString(), item.image, params))
            {
                throw std::runtime_error("failed to write TIFF");
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
            emit queueChanged(queue_.size(), savedCount_);
        }
    }
}

} // namespace ct
