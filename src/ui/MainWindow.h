#pragma once

#include "app/ReconstructionController.h"
#include "core/common/PreviewSliceRenderer.h"
#include "core/config/AppConfig.h"
#include "hardware/camera/PvcamCameraWorker.h"
#include "hardware/kohzu/KohzuController.h"

#include <QMainWindow>
#include <QMap>
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
    void startCameraWorker();
    void snapCameraFrame();
    void toggleCameraLivePreview();
    void stopCameraLivePreview(bool restartAfterStop = false);
    void applyCameraRoi();
    void resetCameraRoiFullFrame();
    void updateCameraFrame(const CameraFrame& frame);
    void renderCameraPreviewWithCurrentWindow();
    void startReconstruction();
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

    ReconstructionController controller_;
    QThread cameraThread_;
    PvcamCameraWorker* cameraWorker_ = nullptr;
    QThread kohzuThread_;
    KohzuController* kohzuController_ = nullptr;
    bool motorConnected_ = false;
    bool shuttingDown_ = false;
    AppConfig sessionConfig_ = AppConfig::defaults();
    cv::Mat lastRawCameraFrame_;
    cv::Mat lastCameraSaveFrame_;
    cv::Mat lastRawReconPreviewSlice_;
    PreviewWindow currentCameraPreviewWindow_;
    PreviewWindow currentReconPreviewWindow_;
    qint64 cameraPreviewFrameCount_ = 0;
    qint64 lastCameraPreviewRenderMs_ = 0;
    int pendingSingleSnapIndex_ = -1;
    bool cameraLiveRunning_ = false;
    bool cameraRestartLiveAfterStop_ = false;
    bool cameraSnapSaveInProgress_ = false;

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
    QComboBox* preprocessMethodCombo_ = nullptr;
    QDoubleSpinBox* airValueSpin_ = nullptr;
    QSpinBox* taperSpin_ = nullptr;
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
    QPushButton* cameraSnapButton_ = nullptr;
    QPushButton* cameraLiveButton_ = nullptr;

    ZoomPanImageView* cameraPreviewView_ = nullptr;
    ZoomPanImageView* reconPreviewView_ = nullptr;
    QSlider* previewSliceSlider_ = nullptr;
    QLabel* previewSliceValueLabel_ = nullptr;
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
    QMap<QString, MotorAxisUi> motorAxes_;
    QMap<int, QString> motorAxisNamesByNumber_;
};

} // namespace ct
