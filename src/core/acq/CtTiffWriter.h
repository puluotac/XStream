#pragma once

#include "core/acq/CtAcquisitionTypes.h"

#include <QMutex>
#include <QObject>
#include <QQueue>
#include <QThread>
#include <QWaitCondition>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace ct
{

class CtTiffWriter : public QObject
{
    Q_OBJECT

public:
    explicit CtTiffWriter(QObject* parent = nullptr);
    ~CtTiffWriter() override;

    void configure(int threadCount, int maxQueuedFrames);

public slots:
    void start();
    void enqueue(const ct::CtFrameSaveItem& item);
    void finish();
    void stop();

signals:
    void queueChanged(int queued, int saved);
    void logMessage(const QString& text);
    void finished(bool ok, const QString& message);

private:
    void workerLoop(int workerIndex);

    QMutex mutex_;
    QWaitCondition notEmpty_;
    QWaitCondition notFull_;
    QQueue<CtFrameSaveItem> queue_;
    int maxQueuedFrames_ = 64;
    int threadCount_ = 16;
    int savedCount_ = 0;
    bool accepting_ = false;
    bool finishRequested_ = false;
    bool stopRequested_ = false;
    bool failed_ = false;
    QString errorMessage_;
    std::vector<std::thread> workers_;
};

} // namespace ct

Q_DECLARE_METATYPE(ct::CtFrameSaveItem)
