#pragma once

#include <QObject>

#include <opencv2/core.hpp>

#include <memory>

namespace ct
{

class IReconstructionVolume;

class PreviewRenderWorker : public QObject
{
    Q_OBJECT

public:
    explicit PreviewRenderWorker(QObject* parent = nullptr);

    void setSource(const std::shared_ptr<IReconstructionVolume>& volume, int depth);
    void clearSource();
    void enqueueRender(int sliceIndex, bool updateHistogram);

signals:
    void previewSliceReady(const cv::Mat& slice, bool updateHistogram);

private:
    void scheduleDrain();
    void drainPendingRequest();

    std::shared_ptr<IReconstructionVolume> volume_;
    int depth_ = 0;
    int pendingSliceIndex_ = -1;
    bool pendingUpdateHistogram_ = false;
    bool renderScheduled_ = false;
};

} // namespace ct
