#include "app/ReconstructionController.h"

#include "core/common/PreviewRenderWorker.h"
#include "core/recon/IReconstructionVolume.h"
#include "core/recon/ReconstructionRuntimeCache.h"
#include "core/recon/ReconstructionWorker.h"

#include <algorithm>

namespace ct
{

namespace
{

bool isOfflineMethod(ReconMethod method)
{
    return method == ReconMethod::OfflineFBP || method == ReconMethod::OfflineGridrec;
}

} // namespace

ReconstructionController::ReconstructionController(QObject* parent)
    : QObject(parent)
{
    runtimeCache_ = std::make_shared<ReconstructionRuntimeCache>();

    previewThread_ = new QThread(this);
    previewWorker_ = new PreviewRenderWorker();
    previewWorker_->moveToThread(previewThread_);

    connect(previewWorker_, &PreviewRenderWorker::previewSliceReady, this, &ReconstructionController::previewSliceReady);

    previewThread_->start();
}

ReconstructionController::~ReconstructionController()
{
    shutdown();
}

void ReconstructionController::shutdown()
{
    stopReconstruction();
    cleanupReconstructionThread();
    cleanupSaveThread();
    cleanupPreviewThread();
}

void ReconstructionController::startReconstruction(const AppConfig& config)
{
    if (busy_)
    {
        return;
    }

    latestVolume_.reset();
    latestConfig_ = config;
    busy_ = true;
    emit busyChanged(true);
    if (previewWorker_ != nullptr)
    {
        QMetaObject::invokeMethod(previewWorker_, [worker = previewWorker_]() {
            worker->clearSource();
        }, Qt::QueuedConnection);
    }

    reconstructionThread_ = new QThread(this);
    reconstructionWorker_ = new ReconstructionWorker(config,
        isOfflineMethod(config.io.reconMethod) ? nullptr : runtimeCache_);
    reconstructionWorker_->moveToThread(reconstructionThread_);

    connect(reconstructionThread_, &QThread::started, reconstructionWorker_, &ReconstructionWorker::process);
    connect(reconstructionWorker_, &ReconstructionWorker::logMessage, this, &ReconstructionController::logMessage);
    connect(reconstructionWorker_, &ReconstructionWorker::progressUpdated, this, &ReconstructionController::progressUpdated);
    connect(reconstructionWorker_, &ReconstructionWorker::corComputed, this, &ReconstructionController::corComputed);
    connect(reconstructionWorker_, &ReconstructionWorker::volumeGeometryResolved, this, [this](int volW, int volH) {
        latestVolume_ = reconstructionWorker_->volumeBuffer();
        emit volumeGeometryResolved(volW, volH);
        if (previewWorker_ != nullptr && latestVolume_)
        {
            const std::shared_ptr<IReconstructionVolume> volume = latestVolume_;
            QMetaObject::invokeMethod(previewWorker_, [worker = previewWorker_, volume, volH]() {
                worker->setSource(volume, volH);
            }, Qt::QueuedConnection);
        }
    });
    connect(reconstructionWorker_, &ReconstructionWorker::previewRefreshSuggested, this, [this](bool updateHistogram) {
        requestPreviewRender(updateHistogram);
    });
    connect(reconstructionWorker_, &ReconstructionWorker::directPreviewSliceReady,
        this, [this](cv::Mat slice, bool updateHistogram) {
            emit previewSliceReady(std::move(slice), updateHistogram);
        });

    connect(reconstructionWorker_, &ReconstructionWorker::reconstructionFinished, this, [this]() {
        const bool hasActiveWorker = reconstructionWorker_ != nullptr;
        latestVolume_ = hasActiveWorker ? reconstructionWorker_->volumeBuffer() : nullptr;
        latestConfig_ = hasActiveWorker ? reconstructionWorker_->config() : latestConfig_;
        busy_ = false;
        emit busyChanged(false);
        if (latestVolume_)
        {
            requestPreviewRender(true);
        }
        const bool hasVolume = static_cast<bool>(latestVolume_);
        cleanupReconstructionThread();
        emit reconstructionCompleted(hasVolume);
    });

    connect(reconstructionWorker_, &ReconstructionWorker::reconstructionFailed, this, [this](const QString& error) {
        busy_ = false;
        emit busyChanged(false);
        cleanupReconstructionThread();
        emit reconstructionFailed(error);
    });

    reconstructionThread_->start();
}

void ReconstructionController::stopReconstruction()
{
    if (reconstructionWorker_ != nullptr)
    {
        QMetaObject::invokeMethod(reconstructionWorker_, "requestStop", Qt::QueuedConnection);
    }
}

void ReconstructionController::setPreviewSliceIndex(int index)
{
    latestConfig_.preview.liveSliceIndex = index;

    if (reconstructionWorker_ != nullptr)
    {
        reconstructionWorker_->setPreviewSliceIndex(index);
    }

    if (busy_ && !latestConfig_.preview.enableLive)
    {
        return;
    }

    requestPreviewRender(false);
}

void ReconstructionController::startSave(const SaveRequest& request)
{
    if (busy_ || !request.volume)
    {
        return;
    }

    busy_ = true;
    emit busyChanged(true);

    saveThread_ = new QThread(this);
    saveWorker_ = new SaveWorker(request);
    saveWorker_->moveToThread(saveThread_);

    connect(saveThread_, &QThread::started, saveWorker_, &SaveWorker::process);
    connect(saveWorker_, &SaveWorker::logMessage, this, &ReconstructionController::logMessage);
    connect(saveWorker_, &SaveWorker::progressUpdated, this, &ReconstructionController::progressUpdated);
    connect(saveWorker_, &SaveWorker::saveFinished, this, [this]() {
        busy_ = false;
        emit busyChanged(false);
        emit saveCompleted();
        cleanupSaveThread();
    });
    connect(saveWorker_, &SaveWorker::saveFailed, this, [this](const QString& error) {
        busy_ = false;
        emit busyChanged(false);
        emit saveFailed(error);
        cleanupSaveThread();
    });

    saveThread_->start();
}

bool ReconstructionController::isBusy() const
{
    return busy_;
}

bool ReconstructionController::hasVolume() const
{
    return static_cast<bool>(latestVolume_);
}

AppConfig ReconstructionController::lastConfig() const
{
    return latestConfig_;
}

std::shared_ptr<IReconstructionVolume> ReconstructionController::latestVolume() const
{
    return latestVolume_;
}

void ReconstructionController::requestPreviewRender(bool updateHistogram)
{
    if (previewWorker_ == nullptr || !latestVolume_)
    {
        return;
    }

    const int maxSlice = std::max(0, latestVolume_->depth() - 1);
    const int clampedIndex = std::clamp(latestConfig_.preview.liveSliceIndex, 0, maxSlice);
    QMetaObject::invokeMethod(previewWorker_, [worker = previewWorker_, clampedIndex, updateHistogram]() {
        worker->enqueueRender(clampedIndex, updateHistogram);
    }, Qt::QueuedConnection);
}

void ReconstructionController::cleanupReconstructionThread()
{
    if (reconstructionThread_ != nullptr)
    {
        ReconstructionWorker* worker = reconstructionWorker_;
        QThread* guiThread = thread();
        if (worker != nullptr)
        {
            QMetaObject::invokeMethod(worker, [worker, guiThread]() {
                worker->moveToThread(guiThread);
            }, Qt::BlockingQueuedConnection);
        }
        reconstructionThread_->quit();
        reconstructionThread_->wait();
        if (worker != nullptr)
        {
            delete worker;
        }
        delete reconstructionThread_;
        reconstructionWorker_ = nullptr;
        reconstructionThread_ = nullptr;
    }
}

void ReconstructionController::cleanupSaveThread()
{
    if (saveWorker_ != nullptr)
    {
        QMetaObject::invokeMethod(saveWorker_, "requestStop", Qt::QueuedConnection);
    }

    if (saveThread_ != nullptr)
    {
        SaveWorker* worker = saveWorker_;
        QThread* guiThread = thread();
        if (worker != nullptr)
        {
            QMetaObject::invokeMethod(worker, [worker, guiThread]() {
                worker->moveToThread(guiThread);
            }, Qt::BlockingQueuedConnection);
        }
        saveThread_->quit();
        saveThread_->wait();
        if (worker != nullptr)
        {
            delete worker;
        }
        delete saveThread_;
        saveWorker_ = nullptr;
        saveThread_ = nullptr;
    }
}

void ReconstructionController::cleanupPreviewThread()
{
    if (previewWorker_ != nullptr)
    {
        QMetaObject::invokeMethod(previewWorker_, [worker = previewWorker_]() {
            worker->clearSource();
        }, Qt::BlockingQueuedConnection);
    }

    if (previewThread_ != nullptr)
    {
        PreviewRenderWorker* worker = previewWorker_;
        QThread* guiThread = thread();
        if (worker != nullptr)
        {
            QMetaObject::invokeMethod(worker, [worker, guiThread]() {
                worker->moveToThread(guiThread);
            }, Qt::BlockingQueuedConnection);
        }
        previewThread_->quit();
        previewThread_->wait();
        if (worker != nullptr)
        {
            delete worker;
        }
        delete previewThread_;
        previewWorker_ = nullptr;
        previewThread_ = nullptr;
    }
}

} // namespace ct
