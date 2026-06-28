#pragma once

#include "core/config/AppConfig.h"
#include "core/save/SaveWorker.h"

#include <QObject>
#include <QThread>

#include <opencv2/core.hpp>

#include <memory>

namespace ct
{

class ReconstructionWorker;
class IReconstructionVolume;
class PreviewRenderWorker;
class ReconstructionRuntimeCache;

class ReconstructionController : public QObject
{
    Q_OBJECT

public:
    explicit ReconstructionController(QObject* parent = nullptr);
    ~ReconstructionController() override;

    void startReconstruction(const AppConfig& config);
    void stopReconstruction();
    void startSave(const SaveRequest& request);
    void setPreviewSliceIndex(int index);
    void shutdown();

    bool isBusy() const;
    bool hasVolume() const;
    AppConfig lastConfig() const;
    std::shared_ptr<IReconstructionVolume> latestVolume() const;

signals:
    void logMessage(const QString& text);
    void progressUpdated(int current, int total);
    void previewSliceReady(cv::Mat slice, bool updateHistogram);
    void corComputed(double shift);
    void volumeGeometryResolved(int volW, int volH);
    void busyChanged(bool busy);
    void reconstructionCompleted(bool hasVolume);
    void reconstructionFailed(const QString& error);
    void saveCompleted();
    void saveFailed(const QString& error);

private:
    void requestPreviewRender(bool updateHistogram);
    void cleanupReconstructionThread();
    void cleanupSaveThread();
    void cleanupPreviewThread();

    QThread* reconstructionThread_ = nullptr;
    ReconstructionWorker* reconstructionWorker_ = nullptr;
    QThread* previewThread_ = nullptr;
    PreviewRenderWorker* previewWorker_ = nullptr;
    QThread* saveThread_ = nullptr;
    SaveWorker* saveWorker_ = nullptr;
    std::shared_ptr<IReconstructionVolume> latestVolume_;
    std::shared_ptr<ReconstructionRuntimeCache> runtimeCache_;
    AppConfig latestConfig_;
    bool busy_ = false;
};

} // namespace ct
