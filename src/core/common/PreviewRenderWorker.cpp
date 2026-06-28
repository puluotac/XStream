#include "core/common/PreviewRenderWorker.h"

#include "core/recon/IReconstructionVolume.h"

#include <QTimer>

#include <algorithm>

namespace ct
{

PreviewRenderWorker::PreviewRenderWorker(QObject* parent)
    : QObject(parent)
{
}

void PreviewRenderWorker::setSource(const std::shared_ptr<IReconstructionVolume>& volume, int depth)
{
    volume_ = volume;
    depth_ = std::max(0, depth);
    pendingSliceIndex_ = -1;
    pendingUpdateHistogram_ = false;
}

void PreviewRenderWorker::clearSource()
{
    volume_.reset();
    depth_ = 0;
    pendingSliceIndex_ = -1;
    pendingUpdateHistogram_ = false;
    renderScheduled_ = false;
}

void PreviewRenderWorker::enqueueRender(int sliceIndex, bool updateHistogram)
{
    if (!volume_ || depth_ <= 0)
    {
        return;
    }

    pendingSliceIndex_ = std::clamp(sliceIndex, 0, depth_ - 1);
    pendingUpdateHistogram_ = updateHistogram;
    scheduleDrain();
}

void PreviewRenderWorker::scheduleDrain()
{
    if (renderScheduled_)
    {
        return;
    }

    renderScheduled_ = true;
    QTimer::singleShot(0, this, [this]() {
        drainPendingRequest();
    });
}

void PreviewRenderWorker::drainPendingRequest()
{
    if (!volume_ || depth_ <= 0 || pendingSliceIndex_ < 0)
    {
        renderScheduled_ = false;
        return;
    }

    const int sliceIndex = pendingSliceIndex_;
    const bool updateHistogram = pendingUpdateHistogram_;
    pendingSliceIndex_ = -1;
    pendingUpdateHistogram_ = false;

    const cv::Mat slice = volume_->downloadSlice(sliceIndex);
    if (!slice.empty())
    {
        emit previewSliceReady(slice.clone(), updateHistogram);
    }

    if (pendingSliceIndex_ >= 0)
    {
        QTimer::singleShot(0, this, [this]() {
            drainPendingRequest();
        });
        return;
    }

    renderScheduled_ = false;
}

} // namespace ct
