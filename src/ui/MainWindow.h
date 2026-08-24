#pragma once

#include "app/ReconstructionController.h"
#include "core/acq/CtAcquisitionTypes.h"
#include "core/acq/ContinuousFrameRouter.h"
#include "core/acq/CtReplayAcquisitionWorker.h"
#include "core/acq/CtTiffWriter.h"
#include "core/common/PreviewSliceRenderer.h"
#include "core/config/AppConfig.h"
#include "hardware/camera/PvcamCameraWorker.h"
#include "hardware/kohzu/KohzuController.h"

#include <QMainWindow>
#include <QMap>
#include <QStringList>
#include <QThread>

#include <opencv2/core.hpp>

class QLabel;
class QLineEdit;
class QTextEdit;
class QProgressBar;
class QPushButton;
class QSlider;
class QSpinBox;
class QDoubleSpinBox;
class QComboBox;
class QCheckBox;
class QCloseEvent;
class QListWidget;
class QFileSystemWatcher;
class QTimer;
class QTabWidget;

namespace ct
{

class ContrastHistogramWidget;
class ZoomPanImageView;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private:
    enum class CtFrameKind
    {
        Tomo,
        Flat,
        Cor180
    };

    AppConfig configFromUi() const;
    void applyConfig(const AppConfig& config);
    void applyDefaults();
    bool saveSessionConfig();
    QString sessionConfigPath() const;
    void setBusy(bool busy);
    void appendLog(const QString& text);
    void closeEvent(QCloseEvent* event) override;
    void shutdownWorkers();

    void browseWatchDir();
    void browseSaveDir();
    void browseCameraSaveDir();
    void browseCtSavePath();
    void startCameraWorker();
    void snapCameraFrame();
    void toggleCameraLivePreview();
    void stopCameraLivePreview(bool restartAfterStop = false);
    void applyCameraRoi();
    void resetCameraRoiFullFrame();
    void updateCameraFrame(const CameraFrame& frame);
    void updateCameraFrameForCtPreview(const CameraFrame& frame);
    void renderCameraPreviewWithCurrentWindow();
    void updateCameraPixelHover(int previewX, int previewY);
    void updateReconPixelHover(int x, int y);
    void startReconstruction();
    void startCtAcquisition();
    void stopCtAcquisition();
    void maybeStartCtSmokeAutostart();
    void maybeStartCtReplayAutostart();
    void startInternalCtReplay();
    void handleReplayCtFrame(const CameraFrame& frame, const QString& filePath, bool isFlat, bool isCor180);
    void finishInternalCtReplay(bool ok, const QString& message);
    void startCtInitialMotorMove();
    void startCtFlyInitialMove();
    void advanceCtFlyAcquisition();
    void startCtFlyScanMove();
    void startCtFlyScanMotorSweep();
    void startCtTimeSeriesCapture();
    void startCtMotorSpeedTableSetup();
    void sendNextCtMotorSpeedTableSetup();
    void finishCtMotorSpeedTableSetup();
    void startCtMotorTiming(const QString& action);
    void finishCtMotorTiming(const QString& action);
    bool moveCtAxisAbsolute(const QString& axisName, double targetValue, const QString& action, const QString& label);
    bool moveCtAxisRelative(const QString& axisName, double deltaValue, const QString& action, const QString& label);
    void advanceCtAcquisition();
    void requestCtFrame(CtFrameKind kind);
    void enqueueCtFrame(const CameraFrame& frame);
    void handleContinuousFrameBatch(const QVector<ContinuousFrameMetadata>& frames, bool ok, const QString& message);
    void finishCtAcquisition(bool ok, const QString& message);
    void writeCtManifest(bool completed, const QString& errorMessage);
    void enqueueCtReconJob();
    void enqueueCurrentReconJob();
    QString reconQueueRoot() const;
    void ensureReconQueueDirs();
    void recoverStaleRunningReconJobs();
    void scheduleReconQueueRefresh();
    void refreshReconQueue();
    void normalizePendingReconQueue(bool scheduleRefresh = true);
    void importReconJobs();
    void moveSelectedReconQueueJob(int direction);
    void removeSelectedReconQueueJob();
    void startNextQueuedReconstruction();
    bool startQueuedReconstruction(const QString& jobPath);
    void finishQueuedReconstruction(bool ok, const QString& message, const QString& failedStage = QString(), const QString& errorCode = QString());
    void updateActiveQueueCor(double shift);
    QString writeReconResultJson(const QString& jobPath, const QJsonObject& job, const QString& status,
        const QString& failedStage, const QString& errorCode, const QString& message,
        const QString& projectionDir, const QString& outputDir, const QString& startedAt,
        const QString& finishedAt, double initialCor, double autoCor, bool hasAutoCor, double finalCor);
    void stopReconstruction();
    void saveSlices();
    void updatePreviewSlice(const cv::Mat& slice, bool updateHistogram);
    void renderPreviewWithCurrentWindow();
    void setPreviewSliceRange(int depth);
    void syncPreviewSliceControls(int value);
    void requestPreviewSlice(int value);
    void updatePreviewSliceLabel(int value);
    void updateReconMethodUiState();
    void autoSelectZChunk();
    void startMotorWorker();
    void configureMotorAxes();
    void requestMotorConnect();
    void updateMotorAxisPosition(int axis, qint64 positionPulse, const QString& statusRaw);
    void updateMotorAxisActive(int axis, bool active);
    void moveMotorAxis(const QString& name);
    void moveMotorAxisRelative(const QString& name, int direction);
    void stopMotorAxis(const QString& name);
    void stopAllMotorAxes();
    void toggleMotorSpeedMode();
    void updateMotorUiEnabled();
    void updateCtModeUiState();
    void startTimingCalibration();
    void handleTimingCalibrationResult(const PvcamTimingCalibrationResult& result);
    void invalidateTimingCalibration(const QString& reason);
    bool timingCalibrationMatchesCurrentUi() const;
    PvcamTimingCalibrationSettings timingCalibrationSettingsFromUi() const;
    void updateCtFlyDerivedParams();
    void updateCtTimeSeriesDerivedParams();
    void updateContinuousTimingUi();
    void updateCtProgressUi();
    int estimateCtTotalFrames() const;
    double estimateAxisMoveSeconds(const QString& axisName, double distanceUnits) const;
    double estimateCtAcquisitionSeconds(int totalFrames) const;

    ReconstructionController controller_;
    QThread cameraThread_;
    PvcamCameraWorker* cameraWorker_ = nullptr;
    QThread ctWriterThread_;
    CtTiffWriter* ctWriter_ = nullptr;
    ContinuousFrameRouter* ctContinuousRouter_ = nullptr;
    QThread ctReplayThread_;
    CtReplayAcquisitionWorker* ctReplayWorker_ = nullptr;
    QThread kohzuThread_;
    KohzuController* kohzuController_ = nullptr;
    bool motorConnected_ = false;
    bool shuttingDown_ = false;
    AppConfig sessionConfig_ = AppConfig::defaults();
    cv::Mat lastRawCameraFrame_;
    cv::Mat lastCameraSaveFrame_;
    cv::Mat lastCameraHoverFrame_;
    cv::Mat lastRawReconPreviewSlice_;
    PreviewWindow currentCameraPreviewWindow_;
    PreviewWindow currentReconPreviewWindow_;
    qint64 cameraPreviewFrameCount_ = 0;
    qint64 lastCameraPreviewRenderMs_ = 0;
    double cameraPreviewScaleX_ = 1.0;
    double cameraPreviewScaleY_ = 1.0;
    int pendingSingleSnapIndex_ = -1;
    bool cameraLiveRunning_ = false;
    bool cameraRestartLiveAfterStop_ = false;
    bool ctRestoreLiveAfterFinish_ = false;
    int liveRestoreGeneration_ = 0;
    bool cameraWarmOpenComplete_ = false;
    qint64 cameraWarmOpenCompletedMs_ = 0;
    bool reconStartDeferredForCameraWarmOpen_ = false;
    bool cameraSnapSaveInProgress_ = false;
    bool ctRunning_ = false;
    bool ctStopRequested_ = false;
    bool ctWaitingForFrame_ = false;
    bool ctWaitingForCaptureFinish_ = false;
    bool ctWaitingForMotor_ = false;
    bool ctPendingFrameIsFlat_ = false;
    CtFrameKind ctPendingFrameKind_ = CtFrameKind::Tomo;
    int ctPendingFrameBatchRemaining_ = 0;
    bool ctProjectionNeedsMove_ = false;
    bool ctCorReferencePending_ = false;
    bool ctCorReferenceCaptured_ = false;
    bool ctYIsOutForFlat_ = false;
    bool ctFinalFlatDone_ = false;
    bool ctReplayRunning_ = false;
    QString ctPendingMotorAction_;
    int ctProjectionIndex_ = 0;
    int ctFlatIndex_ = 0;
    int ctFlatFramesRemaining_ = 0;
    int ctFlatAfterProjectionCount_ = 0;
    int ctLastFlatAfterProjectionCount_ = -1;
    double ctCurrentAngleDeg_ = 0.0;
    QString ctManifestPath_;
    QVector<QString> ctReplayJobQueue_;
    QVector<QString> ctReplayTargetQueue_;
    QVector<QString> ctReplayOutputQueue_;
    QString activeQueueJobPath_;
    QString activeQueueProjectionDir_;
    QString activeQueueOutputDir_;
    QString activeQueueStartedAt_;
    QString activeQueueFinishedStatusOverride_;
    QString activeQueueFinishedStageOverride_;
    QString activeQueueFinishedCodeOverride_;
    double activeQueueInitialCorShift_ = 0.0;
    double activeQueueAutoCorShift_ = 0.0;
    bool activeQueueHasAutoCorShift_ = false;
    QStringList recentLogLines_;
    CtAcquisitionSettings ctSettings_;
    QVector<CtFlatGroup> ctFlatGroups_;
    QVector<QJsonObject> ctFlyFrames_;
    QJsonObject ctCor180Frame_;
    QVector<QJsonObject> ctTimeSeriesFrames_;
    bool ctFlyScanRequested_ = false;
    bool ctFlyScanCapturing_ = false;
    bool ctFlyScanMoveDone_ = false;
    bool ctFlyRestartingPreparedAfterScan_ = false;
    bool ctTimeSeriesCapturing_ = false;
    bool ctContinuousStartPending_ = false;
    bool ctContinuousCaptureActive_ = false;
    QString ctContinuousIntervalStrategy_;
    QString ctContinuousIntervalFallbackReason_;
    QString ctPendingFinishErrorMessage_;
    int ctFlyFlatStage_ = 0;
    int ctFlyContinuousFramesReceived_ = 0;
    int ctFlyRawFramePlan_ = 0;
    double ctFlyDerivedScanTimeS_ = 0.0;
    double ctFlyDerivedFrameIntervalMs_ = 0.0;
    qint64 ctFlyScanStartedMs_ = 0;
    qint64 ctFlyScanFinishedMs_ = 0;
    qint64 ctAcquisitionStartedMs_ = 0;
    qint64 ctFirstFrameRequestedMs_ = 0;
    qint64 ctLastFrameEnqueuedMs_ = 0;
    qint64 ctWriterFinishRequestedMs_ = 0;
    qint64 ctPreparedStartedMs_ = 0;
    qint64 ctPreparedReadyMs_ = 0;
    qint64 ctCurrentFrameBatchStartedMs_ = 0;
    qint64 ctCurrentMotorMoveStartedMs_ = 0;
    qint64 ctPreparedSetupMs_ = 0;
    qint64 ctMotorMoveMs_ = 0;
    qint64 ctRMoveMs_ = 0;
    qint64 ctYMoveMs_ = 0;
    qint64 ctFrameCaptureMs_ = 0;
    qint64 ctFrameEnqueueMs_ = 0;
    qint64 ctWriterDrainMs_ = 0;
    qint64 ctCompletedMs_ = 0;
    int ctMotorMoveCount_ = 0;
    int ctRMoveCount_ = 0;
    int ctYMoveCount_ = 0;
    int ctFrameBatchCount_ = 0;
    int ctFrameSavedCount_ = 0;
    int ctContinuousFrameBaseCount_ = 0;
    qint64 ctWriterStartedMs_ = 0;
    qint64 ctWriterFrameBytes_ = 0;
    int ctWriterThreads_ = 0;
    int ctWriterQueueCapacity_ = 0;
    qint64 ctContinuousRawFrames_ = 0;
    qint64 ctContinuousDiscardedFrames_ = 0;
    qint64 ctContinuousDeviceDroppedFrames_ = 0;
    int ctFramePoolInUse_ = 0;
    int ctFramePoolPeak_ = 0;
    ContinuousTimingStats ctContinuousTimingStats_;
    int ctContinuousTimingDriftLevel_ = 0;
    int ctEstimatedTotalFrames_ = 0;
    double ctEstimatedTotalSeconds_ = 0.0;
    double ctFlyDerivedSpeedDegPerSec_ = 1.0;
    double ctFlyDerivedLeadAngleDeg_ = 3.0;
    double ctFlyFirstKeepDelayMs_ = 0.0;
    bool ctMotorSpeedTablesReady_ = false;
    int ctMotorSpeedSetupStage_ = 0;
    bool ctIdlePollingPausedForCt_ = false;
    bool timingCalibrationRunning_ = false;
    bool timingCalibrationValid_ = false;
    bool timingCalibrationPendingAfterLiveStop_ = false;
    bool timingRestoreLiveAfterFinish_ = false;
    QString timingInvalidReason_;
    PvcamTimingCalibrationSettings timingCalibrationSettings_;
    PvcamTimingCalibrationResult timingCalibrationResult_;
    int ctFlyRawFrameStride_ = 1;
    int ctTimeSeriesRawFrameStride_ = 1;
    double ctTimeSeriesActualIntervalMs_ = 0.0;

    QLineEdit* watchDirEdit_ = nullptr;
    QLineEdit* saveDirEdit_ = nullptr;
    QComboBox* correctionModeCombo_ = nullptr;
    QComboBox* reconMethodCombo_ = nullptr;
    QLineEdit* tomoPrefixEdit_ = nullptr;
    QLineEdit* flatPrefixEdit_ = nullptr;
    QLineEdit* darkPrefixEdit_ = nullptr;
    QSpinBox* paddingSpin_ = nullptr;
    QComboBox* extCombo_ = nullptr;
    QLineEdit* file0Edit_ = nullptr;
    QLineEdit* file180Edit_ = nullptr;
    QSpinBox* volWSpin_ = nullptr;
    QSpinBox* volHSpin_ = nullptr;
    QSpinBox* totalAnglesSpin_ = nullptr;
    QCheckBox* corAutoCheck_ = nullptr;
    QDoubleSpinBox* corShiftSpin_ = nullptr;
    QDoubleSpinBox* angleOffsetSpin_ = nullptr;
    QCheckBox* phaseEnableCheck_ = nullptr;
    QDoubleSpinBox* energySpin_ = nullptr;
    QDoubleSpinBox* distanceSpin_ = nullptr;
    QDoubleSpinBox* pixelSpin_ = nullptr;
    QDoubleSpinBox* alphaSpin_ = nullptr;
    QCheckBox* stripeEnableCheck_ = nullptr;
    QSpinBox* stripeLevelSpin_ = nullptr;
    QDoubleSpinBox* stripeSigmaSpin_ = nullptr;
    QSpinBox* taperSpin_ = nullptr;
    QSpinBox* flatGapSpin_ = nullptr;
    QSpinBox* flatNumSpin_ = nullptr;
    QComboBox* filterCombo_ = nullptr;
    QSpinBox* batchSpin_ = nullptr;
    QSpinBox* zChunkSpin_ = nullptr;
    QPushButton* autoZChunkButton_ = nullptr;
    QSpinBox* gpuSpin_ = nullptr;
    QSpinBox* zBlockSpin_ = nullptr;
    QCheckBox* livePreviewCheck_ = nullptr;
    QSpinBox* liveSliceSpin_ = nullptr;
    QComboBox* saveDepthCombo_ = nullptr;
    QComboBox* saveFormatCombo_ = nullptr;
    QSpinBox* tiffThreadsSpin_ = nullptr;
    QLineEdit* savePrefixEdit_ = nullptr;
    QListWidget* reconQueueList_ = nullptr;
    QPushButton* reconQueueUpButton_ = nullptr;
    QPushButton* reconQueueDownButton_ = nullptr;
    QPushButton* reconQueueRemoveButton_ = nullptr;
    QPushButton* reconQueueCurrentButton_ = nullptr;
    QPushButton* reconQueueImportButton_ = nullptr;
    QFileSystemWatcher* reconQueueWatcher_ = nullptr;
    QTimer* reconQueueRefreshTimer_ = nullptr;
    QTimer* reconQueuePollTimer_ = nullptr;

    QDoubleSpinBox* cameraExposureSpin_ = nullptr;
    QComboBox* cameraExposureUnitCombo_ = nullptr;
    QSpinBox* cameraRoiXSpin_ = nullptr;
    QSpinBox* cameraRoiYSpin_ = nullptr;
    QSpinBox* cameraRoiWidthSpin_ = nullptr;
    QSpinBox* cameraRoiHeightSpin_ = nullptr;
    QPushButton* cameraApplyRoiButton_ = nullptr;
    QPushButton* cameraResetRoiButton_ = nullptr;
    QLineEdit* cameraSnapPrefixEdit_ = nullptr;
    QSpinBox* cameraSnapPaddingSpin_ = nullptr;
    QSpinBox* cameraSnapIndexSpin_ = nullptr;
    QLineEdit* cameraSaveDirEdit_ = nullptr;
    QLabel* cameraStateLabel_ = nullptr;
    QLabel* cameraPixelLabel_ = nullptr;
    QPushButton* cameraSnapButton_ = nullptr;
    QPushButton* cameraLiveButton_ = nullptr;
    QDoubleSpinBox* ctBeginningAngleSpin_ = nullptr;
    QTabWidget* ctModeTabs_ = nullptr;
    QLineEdit* ctSavePathEdit_ = nullptr;
    QSpinBox* ctProjectionNumSpin_ = nullptr;
    QSpinBox* ctFlatGapSpin_ = nullptr;
    QDoubleSpinBox* ctFlatMoveSpin_ = nullptr;
    QSpinBox* ctFlatNumSpin_ = nullptr;
    QDoubleSpinBox* ctFlyAllowedBlurSpin_ = nullptr;
    QDoubleSpinBox* ctFlyTotalAngleSpin_ = nullptr;
    QLabel* ctTimingStatusValueLabel_ = nullptr;
    QLabel* ctTimingSourceValueLabel_ = nullptr;
    QLabel* ctTimingExposureValueLabel_ = nullptr;
    QLabel* ctTimingRoiValueLabel_ = nullptr;
    QLabel* ctTimingReadoutValueLabel_ = nullptr;
    QLabel* ctTimingSamplesValueLabel_ = nullptr;
    QLabel* ctTimingPeriodValueLabel_ = nullptr;
    QLabel* ctTimingRateValueLabel_ = nullptr;
    QLabel* ctTimingMaxDeviationValueLabel_ = nullptr;
    QLabel* ctTimingRangeValueLabel_ = nullptr;
    QLabel* ctTimingDriftValueLabel_ = nullptr;
    QLabel* ctTimingFitResidualValueLabel_ = nullptr;
    QLabel* ctTimingExposureRangeValueLabel_ = nullptr;
    QLabel* ctTimingFramesValueLabel_ = nullptr;
    QLabel* ctTimingDroppedValueLabel_ = nullptr;
    QPushButton* ctTimingCalibrateButton_ = nullptr;
    QLabel* ctFlyRadiusLabel_ = nullptr;
    QLabel* ctFlySpeedLabel_ = nullptr;
    QLabel* ctFlyLeadAngleLabel_ = nullptr;
    QLabel* ctFlyScanTimeLabel_ = nullptr;
    QLabel* ctFlyFrameIntervalLabel_ = nullptr;
    QLabel* ctFlyRawStrideLabel_ = nullptr;
    QLabel* ctFlyValidationLabel_ = nullptr;
    QSpinBox* ctTimeSeriesRawFrameStrideSpin_ = nullptr;
    QLabel* ctTimeSeriesActualIntervalLabel_ = nullptr;
    QLabel* ctTimeSeriesActualRateLabel_ = nullptr;
    QLabel* ctTimeSeriesDurationLabel_ = nullptr;
    QLabel* ctTimeSeriesDataSizeLabel_ = nullptr;
    QLabel* ctTimeSeriesValidationLabel_ = nullptr;
    QSpinBox* ctTiffThreadsSpin_ = nullptr;
    QLabel* ctStatusLabel_ = nullptr;
    QLabel* ctTimingLabel_ = nullptr;
    QProgressBar* ctProgressBar_ = nullptr;
    QLabel* ctWriterLabel_ = nullptr;
    QLabel* ctBufferLabel_ = nullptr;
    QLabel* ctFrameTimingLabel_ = nullptr;
    QPushButton* ctStartButton_ = nullptr;
    QPushButton* ctStopButton_ = nullptr;

    ZoomPanImageView* cameraPreviewView_ = nullptr;
    ZoomPanImageView* reconPreviewView_ = nullptr;
    QSlider* previewSliceSlider_ = nullptr;
    QLabel* previewSliceValueLabel_ = nullptr;
    QLabel* reconPixelLabel_ = nullptr;
    ContrastHistogramWidget* cameraHistogramWidget_ = nullptr;
    ContrastHistogramWidget* reconHistogramWidget_ = nullptr;
    bool cameraPreviewFitInitialized_ = false;
    bool reconPreviewFitInitialized_ = false;
    QTextEdit* logEdit_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QPushButton* startButton_ = nullptr;
    QPushButton* stopButton_ = nullptr;
    QPushButton* saveButton_ = nullptr;

    struct MotorAxisUi
    {
        QString name;
        int axis = 0;
        double pulsePerUnit = 1.0;
        QString unit;
        double currentValue = 0.0;
        bool hasPosition = false;
        double softMin = 0.0;
        double softMax = 0.0;
        QLabel* nameLabel = nullptr;
        QLabel* positionLabel = nullptr;
        QLabel* statusLabel = nullptr;
        QLabel* unitLabel = nullptr;
        QWidget* relativeWidget = nullptr;
        QPushButton* relativeMinusButton = nullptr;
        QDoubleSpinBox* relativeSpin = nullptr;
        QPushButton* relativePlusButton = nullptr;
        QDoubleSpinBox* absoluteSpin = nullptr;
        QPushButton* goButton = nullptr;
        QLabel* speedUnitLabel = nullptr;
        QWidget* speedWidget = nullptr;
        QDoubleSpinBox* baseSpeedSpin = nullptr;
        QDoubleSpinBox* speedSpin = nullptr;
        QDoubleSpinBox* accelSpin = nullptr;
        bool active = false;
    };

    QLineEdit* motorHostEdit_ = nullptr;
    QSpinBox* motorPortSpin_ = nullptr;
    QLabel* motorConnectionLabel_ = nullptr;
    QWidget* motorMotionHeader_ = nullptr;
    QWidget* motorSpeedHeader_ = nullptr;
    QPushButton* motorModeButton_ = nullptr;
    QPushButton* motorStopAllButton_ = nullptr;
    bool motorSpeedMode_ = false;
    int ctSmokeAutostartAttempts_ = 0;
    bool ctSmokeParametersApplied_ = false;
    bool ctSmokeWaitingForTiming_ = false;
    bool ctSmokeAcquisitionStarted_ = false;
    int ctSmokeTargetTab_ = -1;
    QTimer* ctProgressTimer_ = nullptr;
    QMap<QString, MotorAxisUi> motorAxes_;
    QMap<int, QString> motorAxisNamesByNumber_;
};

} // namespace ct
