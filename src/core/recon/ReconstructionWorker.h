#pragma once

#include "core/common/ProjectionTypes.h"
#include "core/config/AppConfig.h"

#include <QObject>

#include <atomic>
#include <memory>
#include <vector>

namespace ct
{

class IReconstructionVolume;
class ReconstructionRuntimeCache;

class ReconstructionWorker : public QObject
{
    Q_OBJECT

public:
    explicit ReconstructionWorker(AppConfig config,
        std::shared_ptr<ReconstructionRuntimeCache> runtimeCache,
        QObject* parent = nullptr);

    std::shared_ptr<IReconstructionVolume> volumeBuffer() const;
    AppConfig config() const;
    void setPreviewSliceIndex(int index);

public slots:
    void process();
    void requestStop();

signals:
    void logMessage(const QString& text);
    void progressUpdated(int current, int total);
    void previewRefreshSuggested(bool updateHistogram);
    void directPreviewSliceReady(const cv::Mat& slice, bool updateHistogram);
    void corComputed(double shift);
    void volumeGeometryResolved(int volW, int volH);
    void reconstructionFinished();
    void reconstructionFailed(const QString& error);

private:
    AppConfig config_;
    std::atomic_bool stopRequested_ = false;
    std::atomic_int previewSliceIndex_ = 0;
    std::shared_ptr<IReconstructionVolume> volumeBuffer_;
    std::shared_ptr<ReconstructionRuntimeCache> runtimeCache_;
};

} // namespace ct
