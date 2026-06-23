#include "ui/MainWindow.h"

#include "core/common/PreviewSliceRenderer.h"
#include "ui/ContrastHistogramWidget.h"
#include "ui/ZoomPanImageView.h"

#include <QCloseEvent>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace ct
{

namespace
{

template <typename T>
T* makeSpin(QWidget* parent, double min, double max, double step)
{
    auto* spin = new T(parent);
    spin->setRange(min, max);
    spin->setSingleStep(step);
    return spin;
}

struct MotorAxisDefinition
{
    const char* name;
    const char* function;
    const char* model;
    int axis;
    const char* channel;
    const char* unit;
    double pulsePerUnit;
    double defaultStep;
    double maxStep;
};

const MotorAxisDefinition kMotorAxes[] = {
    { "X90", "parallel sample trim", "XA04A-R102", 16, "CH8-2", "mm", 20000.0, 0.01, 0.5 },
    { "X0", "vertical sample trim", "XA04A-R102", 15, "CH8-1", "mm", 20000.0, 0.01, 0.5 },
    { "R", "CT rotation", "RA10A-W01", 17, "CH9-1", "deg", 10000.0, 0.01, 1.0 },
    { "U", "goniometer vertical beam", "SA10A-RM01", 1, "CH1-1", "deg", 11904.7619, 0.001, 0.1 },
    { "V", "goniometer parallel beam", "SA10A-RB01", 2, "CH1-2", "deg", 14925.3731, 0.001, 0.1 },
    { "Z", "sample height", "ZA16A-32F01", 18, "CH9-2", "mm", 100000.0, 0.001, 0.1 },
    { "Y", "large horizontal stage", "XA16F-L2101", 3, "CH2-1", "mm", 2000.0, 0.01, 1.0 },
    { "Det_OT2_Z", "Kinetix detector vertical lift", "ZA16A-32F01", 22, "CH11-2", "mm", 100000.0, 0.001, 0.1 },
    { "Det_OT2_Y", "Kinetix detector horizontal travel", "XA16F-L2301", 9, "CH5-1", "mm", 2000.0, 0.01, 1.0 },
};

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("C++ CT Workstation v1"));
    resize(1850, 980);
    cameraPreviewFitInitialized_ = false;
    reconPreviewFitInitialized_ = false;

    auto* central = new QWidget(this);
    auto* mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(8, 8, 8, 8);
    mainLayout->setSpacing(8);
    setCentralWidget(central);

    auto* hardwareScroll = new QScrollArea(central);
    hardwareScroll->setWidgetResizable(true);
    hardwareScroll->setFixedWidth(315);
    auto* hardwareHost = new QWidget(hardwareScroll);
    auto* hardwareLayout = new QVBoxLayout(hardwareHost);
    hardwareLayout->setContentsMargins(6, 6, 6, 6);
    hardwareLayout->setSpacing(6);

    auto* cameraGroup = new QGroupBox(QStringLiteral("Kinetix Camera"), hardwareHost);
    auto* cameraForm = new QFormLayout(cameraGroup);
    cameraForm->setContentsMargins(8, 8, 8, 8);
    cameraForm->setHorizontalSpacing(6);
    cameraForm->setVerticalSpacing(4);
    cameraForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    cameraStateLabel_ = new QLabel(QStringLiteral("Idle"), cameraGroup);
    cameraStatsLabel_ = new QLabel(QStringLiteral("cap 0 / q 0 / use 0 / drop 0"), cameraGroup);
    cameraExposureSpin_ = makeSpin<QSpinBox>(cameraGroup, 1, 600000, 1);
    cameraExposureSpin_->setValue(10);
    cameraExposureSpin_->setSuffix(QStringLiteral(" ms"));
    cameraRoiWidthSpin_ = makeSpin<QSpinBox>(cameraGroup, 1, 10000, 1);
    cameraRoiWidthSpin_->setValue(512);
    cameraRoiHeightSpin_ = makeSpin<QSpinBox>(cameraGroup, 1, 10000, 1);
    cameraRoiHeightSpin_->setValue(512);
    cameraFrameCountSpin_ = makeSpin<QSpinBox>(cameraGroup, 1, 1000000, 1);
    cameraFrameCountSpin_->setValue(10);
    cameraQueueCapacitySpin_ = makeSpin<QSpinBox>(cameraGroup, 1, 100000, 1);
    cameraQueueCapacitySpin_->setValue(256);
    cameraFullFrameCheck_ = new QCheckBox(QStringLiteral("Full Frame"), cameraGroup);
    cameraSaveRawCheck_ = new QCheckBox(QStringLiteral("Save TIFF"), cameraGroup);
    cameraTriggerModeCombo_ = new QComboBox(cameraGroup);
    cameraTriggerModeCombo_->addItems({ "Internal", "External (reserved)" });
    cameraTriggerModeCombo_->setCurrentText(QStringLiteral("Internal"));
    cameraSaveDirEdit_ = new QLineEdit(QStringLiteral("E:/2026_Chun/2026_06_09/pvcam_capture_test"), cameraGroup);
    cameraSaveDirEdit_->setMinimumWidth(120);
    auto* cameraSaveBrowse = new QPushButton(QStringLiteral("..."), cameraGroup);
    connect(cameraSaveBrowse, &QPushButton::clicked, this, &MainWindow::browseCameraSaveDir);
    auto* cameraSaveLayout = new QHBoxLayout();
    cameraSaveLayout->addWidget(cameraSaveDirEdit_);
    cameraSaveLayout->addWidget(cameraSaveBrowse);
    auto* cameraSaveWidget = new QWidget(cameraGroup);
    cameraSaveWidget->setLayout(cameraSaveLayout);
    cameraCaptureButton_ = new QPushButton(QStringLiteral("Capture"), cameraGroup);
    cameraLiveButton_ = new QPushButton(QStringLiteral("Live"), cameraGroup);
    cameraStopButton_ = new QPushButton(QStringLiteral("Stop"), cameraGroup);
    cameraStopButton_->setEnabled(false);
    auto* cameraButtonLayout = new QHBoxLayout();
    cameraButtonLayout->setContentsMargins(0, 0, 0, 0);
    cameraButtonLayout->addWidget(cameraCaptureButton_);
    cameraButtonLayout->addWidget(cameraLiveButton_);
    cameraButtonLayout->addWidget(cameraStopButton_);
    auto* cameraButtonWidget = new QWidget(cameraGroup);
    cameraButtonWidget->setLayout(cameraButtonLayout);
    cameraForm->addRow(QStringLiteral("State"), cameraStateLabel_);
    cameraForm->addRow(QStringLiteral("Stats"), cameraStatsLabel_);
    cameraForm->addRow(QStringLiteral("Exposure"), cameraExposureSpin_);
    cameraForm->addRow(QStringLiteral("ROI W"), cameraRoiWidthSpin_);
    cameraForm->addRow(QStringLiteral("ROI H"), cameraRoiHeightSpin_);
    cameraForm->addRow(QStringLiteral("Frames"), cameraFrameCountSpin_);
    cameraForm->addRow(QStringLiteral("Queue"), cameraQueueCapacitySpin_);
    cameraForm->addRow(cameraFullFrameCheck_);
    cameraForm->addRow(QStringLiteral("Trigger"), cameraTriggerModeCombo_);
    cameraForm->addRow(cameraSaveRawCheck_);
    cameraForm->addRow(QStringLiteral("Save Dir"), cameraSaveWidget);
    cameraForm->addRow(cameraButtonWidget);
    hardwareLayout->addWidget(cameraGroup);

    auto* motorConnectGroup = new QGroupBox(QStringLiteral("Kohzu ARIES"), hardwareHost);
    auto* motorConnectForm = new QFormLayout(motorConnectGroup);
    motorConnectForm->setContentsMargins(8, 8, 8, 8);
    motorConnectForm->setHorizontalSpacing(6);
    motorConnectForm->setVerticalSpacing(4);
    motorHostEdit_ = new QLineEdit(QStringLiteral("10.30.92.201"), motorConnectGroup);
    motorPortSpin_ = makeSpin<QSpinBox>(motorConnectGroup, 1, 65535, 1);
    motorPortSpin_->setValue(12321);
    motorConnectionLabel_ = new QLabel(QStringLiteral("Disconnected"), motorConnectGroup);
    motorConnectForm->addRow(QStringLiteral("Host"), motorHostEdit_);
    motorConnectForm->addRow(QStringLiteral("Port"), motorPortSpin_);
    motorConnectForm->addRow(QStringLiteral("State"), motorConnectionLabel_);
    hardwareLayout->addWidget(motorConnectGroup);

    auto* axisGroup = new QGroupBox(QStringLiteral("Motor Axes"), hardwareHost);
    auto* axisLayout = new QGridLayout(axisGroup);
    axisLayout->setContentsMargins(8, 8, 8, 8);
    axisLayout->setHorizontalSpacing(4);
    axisLayout->setVerticalSpacing(4);
    axisLayout->addWidget(new QLabel(QStringLiteral("Axis"), axisGroup), 0, 0);
    axisLayout->addWidget(new QLabel(QStringLiteral("Position"), axisGroup), 0, 1);
    axisLayout->addWidget(new QLabel(QStringLiteral("Step"), axisGroup), 0, 2);
    axisLayout->addWidget(new QLabel(QStringLiteral("Move"), axisGroup), 0, 3);
    axisLayout->addWidget(new QLabel(QStringLiteral("Stop"), axisGroup), 0, 4);

    int axisRow = 1;
    for (const MotorAxisDefinition& def : kMotorAxes)
    {
        MotorAxisUi axisUi;
        axisUi.name = QString::fromLatin1(def.name);
        axisUi.axis = def.axis;
        axisUi.pulsePerUnit = def.pulsePerUnit;
        axisUi.unit = QString::fromLatin1(def.unit);

        auto* axisName = new QLabel(QStringLiteral("%1\nA%2 %3").arg(axisUi.name).arg(def.axis).arg(QString::fromLatin1(def.channel)), axisGroup);
        axisName->setToolTip(QStringLiteral("%1\n%2").arg(QString::fromLatin1(def.model), QString::fromLatin1(def.function)));
        axisUi.positionLabel = new QLabel(QStringLiteral("-"), axisGroup);
        axisUi.positionLabel->setMinimumWidth(68);
        axisUi.statusLabel = new QLabel(QStringLiteral(""), axisGroup);
        axisUi.statusLabel->setVisible(false);
        axisUi.stepSpin = makeSpin<QDoubleSpinBox>(axisGroup, 0.0, def.maxStep, def.defaultStep);
        axisUi.stepSpin->setDecimals(4);
        axisUi.stepSpin->setValue(def.defaultStep);
        axisUi.stepSpin->setSuffix(QStringLiteral(" %1").arg(axisUi.unit));
        axisUi.stepSpin->setMaximumWidth(84);
        axisUi.minusButton = new QPushButton(QStringLiteral("-"), axisGroup);
        axisUi.plusButton = new QPushButton(QStringLiteral("+"), axisGroup);
        axisUi.stopButton = new QPushButton(QStringLiteral("Stop"), axisGroup);
        axisUi.minusButton->setFixedWidth(28);
        axisUi.plusButton->setFixedWidth(28);
        axisUi.stopButton->setFixedWidth(48);

        auto* moveLayout = new QHBoxLayout();
        moveLayout->setContentsMargins(0, 0, 0, 0);
        moveLayout->addWidget(axisUi.minusButton);
        moveLayout->addWidget(axisUi.plusButton);
        auto* moveWidget = new QWidget(axisGroup);
        moveWidget->setLayout(moveLayout);

        axisLayout->addWidget(axisName, axisRow, 0);
        axisLayout->addWidget(axisUi.positionLabel, axisRow, 1);
        axisLayout->addWidget(axisUi.stepSpin, axisRow, 2);
        axisLayout->addWidget(moveWidget, axisRow, 3);
        axisLayout->addWidget(axisUi.stopButton, axisRow, 4);

        const QString axisKey = axisUi.name;
        connect(axisUi.minusButton, &QPushButton::clicked, this, [this, axisKey]() {
            jogMotorAxis(axisKey, -1);
        });
        connect(axisUi.plusButton, &QPushButton::clicked, this, [this, axisKey]() {
            jogMotorAxis(axisKey, 1);
        });
        connect(axisUi.stopButton, &QPushButton::clicked, this, [this, axisKey]() {
            stopMotorAxis(axisKey);
        });

        motorAxes_.insert(axisKey, axisUi);
        motorAxisNamesByNumber_.insert(axisUi.axis, axisKey);
        ++axisRow;
    }
    axisLayout->setColumnStretch(2, 1);
    hardwareLayout->addWidget(axisGroup);
    hardwareLayout->addStretch();
    hardwareScroll->setWidget(hardwareHost);
    mainLayout->addWidget(hardwareScroll);

    auto* scroll = new QScrollArea(central);
    scroll->setWidgetResizable(true);
    scroll->setFixedWidth(300);
    auto* settingsHost = new QWidget(scroll);
    auto* settingsLayout = new QVBoxLayout(settingsHost);
    settingsLayout->setContentsMargins(6, 6, 6, 6);
    settingsLayout->setSpacing(6);

    auto* ioGroup = new QGroupBox(QStringLiteral("IO"), settingsHost);
    auto* ioForm = new QFormLayout(ioGroup);
    watchDirEdit_ = new QLineEdit(ioGroup);
    auto* watchBrowse = new QPushButton(QStringLiteral("..."), ioGroup);
    connect(watchBrowse, &QPushButton::clicked, this, &MainWindow::browseWatchDir);
    auto* watchLayout = new QHBoxLayout();
    watchLayout->addWidget(watchDirEdit_);
    watchLayout->addWidget(watchBrowse);
    auto* watchWidget = new QWidget(ioGroup);
    watchWidget->setLayout(watchLayout);
    ioForm->addRow(QStringLiteral("Watch Dir"), watchWidget);
    saveDirEdit_ = new QLineEdit(ioGroup);
    auto* saveBrowseBtn = new QPushButton(QStringLiteral("..."), ioGroup);
    connect(saveBrowseBtn, &QPushButton::clicked, this, &MainWindow::browseSaveDir);
    auto* saveLayout = new QHBoxLayout();
    saveLayout->addWidget(saveDirEdit_);
    saveLayout->addWidget(saveBrowseBtn);
    auto* saveWidget = new QWidget(ioGroup);
    saveWidget->setLayout(saveLayout);
    ioForm->addRow(QStringLiteral("Save Dir"), saveWidget);
    correctionModeCombo_ = new QComboBox(ioGroup);
    correctionModeCombo_->addItems({ "None", "Flat", "Flat-Dark" });
    ioForm->addRow(QStringLiteral("Correction"), correctionModeCombo_);
    reconMethodCombo_ = new QComboBox(ioGroup);
    reconMethodCombo_->addItems({ "RealTimeFBP", "OfflineFBP", "OfflineGridrec" });
    ioForm->addRow(QStringLiteral("Recon Method"), reconMethodCombo_);
    connect(reconMethodCombo_, &QComboBox::currentTextChanged,
        this, &MainWindow::updateReconMethodUiState);
    tomoPrefixEdit_ = new QLineEdit(ioGroup);
    flatPrefixEdit_ = new QLineEdit(ioGroup);
    darkPrefixEdit_ = new QLineEdit(ioGroup);
    ioForm->addRow(QStringLiteral("Tomo Prefix"), tomoPrefixEdit_);
    ioForm->addRow(QStringLiteral("Flat Prefix"), flatPrefixEdit_);
    ioForm->addRow(QStringLiteral("Dark Prefix"), darkPrefixEdit_);
    paddingSpin_ = makeSpin<QSpinBox>(ioGroup, 1, 10, 1);
    extCombo_ = new QComboBox(ioGroup);
    extCombo_->addItems({ "tif", "tiff" });
    ioForm->addRow(QStringLiteral("Padding"), paddingSpin_);
    ioForm->addRow(QStringLiteral("Ext"), extCombo_);
    file0Edit_ = new QLineEdit(ioGroup);
    file180Edit_ = new QLineEdit(ioGroup);
    ioForm->addRow(QStringLiteral("0 Deg"), file0Edit_);
    ioForm->addRow(QStringLiteral("180 Deg"), file180Edit_);
    savePrefixEdit_ = new QLineEdit(ioGroup);
    saveDepthCombo_ = new QComboBox(ioGroup);
    saveDepthCombo_->addItems({ "8bit", "16bit", "32bit" });
    saveFormatCombo_ = new QComboBox(ioGroup);
    saveFormatCombo_->addItems({ "tif", "tiff" });
    tiffThreadsSpin_ = makeSpin<QSpinBox>(ioGroup, 1, 64, 1);
    ioForm->addRow(QStringLiteral("Save Prefix"), savePrefixEdit_);
    ioForm->addRow(QStringLiteral("Save Depth"), saveDepthCombo_);
    ioForm->addRow(QStringLiteral("Save Format"), saveFormatCombo_);
    ioForm->addRow(QStringLiteral("TIFF Threads"), tiffThreadsSpin_);
    settingsLayout->addWidget(ioGroup);

    auto* geoGroup = new QGroupBox(QStringLiteral("Geometry"), settingsHost);
    auto* geoForm = new QFormLayout(geoGroup);
    volWSpin_ = makeSpin<QSpinBox>(geoGroup, 1, 4096, 1);
    volHSpin_ = makeSpin<QSpinBox>(geoGroup, 1, 8192, 1);
    totalAnglesSpin_ = makeSpin<QSpinBox>(geoGroup, 1, 7200, 1);
    corAutoCheck_ = new QCheckBox(QStringLiteral("Auto COR"), geoGroup);
    corShiftSpin_ = makeSpin<QDoubleSpinBox>(geoGroup, -4000.0, 4000.0, 0.1);
    angleOffsetSpin_ = makeSpin<QDoubleSpinBox>(geoGroup, -180.0, 180.0, 0.1);
    geoForm->addRow(QStringLiteral("Vol W"), volWSpin_);
    geoForm->addRow(QStringLiteral("Vol H"), volHSpin_);
    geoForm->addRow(QStringLiteral("Angles"), totalAnglesSpin_);
    geoForm->addRow(corAutoCheck_);
    geoForm->addRow(QStringLiteral("COR Shift"), corShiftSpin_);
    geoForm->addRow(QStringLiteral("Angle Offset"), angleOffsetSpin_);
    settingsLayout->addWidget(geoGroup);

    auto* phaseGroup = new QGroupBox(QStringLiteral("Phase"), settingsHost);
    auto* phaseForm = new QFormLayout(phaseGroup);
    phaseEnableCheck_ = new QCheckBox(QStringLiteral("Enable Paganin"), phaseGroup);
    energySpin_ = makeSpin<QDoubleSpinBox>(phaseGroup, 0.0, 500.0, 0.1);
    distanceSpin_ = makeSpin<QDoubleSpinBox>(phaseGroup, 0.0, 10.0, 0.001);
    pixelSpin_ = makeSpin<QDoubleSpinBox>(phaseGroup, 0.0, 1000.0, 0.1);
    alphaSpin_ = makeSpin<QDoubleSpinBox>(phaseGroup, 0.0, 10000.0, 1.0);
    phaseForm->addRow(phaseEnableCheck_);
    phaseForm->addRow(QStringLiteral("Energy keV"), energySpin_);
    phaseForm->addRow(QStringLiteral("Distance m"), distanceSpin_);
    phaseForm->addRow(QStringLiteral("Pixel um"), pixelSpin_);
    phaseForm->addRow(QStringLiteral("Alpha Ratio"), alphaSpin_);
    settingsLayout->addWidget(phaseGroup);

    auto* preprocessGroup = new QGroupBox(QStringLiteral("Preprocess"), settingsHost);
    auto* preprocessForm = new QFormLayout(preprocessGroup);
    preprocessMethodCombo_ = new QComboBox(preprocessGroup);
    preprocessMethodCombo_->addItems({ "linear", "log" });
    airValueSpin_ = makeSpin<QDoubleSpinBox>(preprocessGroup, 0.0, 65535.0, 0.1);
    taperSpin_ = makeSpin<QSpinBox>(preprocessGroup, 0, 200, 1);
    filterCombo_ = new QComboBox(preprocessGroup);
    filterCombo_->addItems({ "shepp-logan", "ram-lak", "cosine", "hamming", "hann" });
    preprocessForm->addRow(QStringLiteral("Method"), preprocessMethodCombo_);
    preprocessForm->addRow(QStringLiteral("Air Value"), airValueSpin_);
    preprocessForm->addRow(QStringLiteral("Taper"), taperSpin_);
    preprocessForm->addRow(QStringLiteral("Filter"), filterCombo_);
    settingsLayout->addWidget(preprocessGroup);

    auto* computeGroup = new QGroupBox(QStringLiteral("Compute"), settingsHost);
    auto* computeForm = new QFormLayout(computeGroup);
    batchSpin_ = makeSpin<QSpinBox>(computeGroup, 1, 200, 1);
    zChunkSpin_ = makeSpin<QSpinBox>(computeGroup, 1, 1024, 1);
    autoZChunkButton_ = new QPushButton(QStringLiteral("Auto Z Chunk"), computeGroup);
    gpuSpin_ = makeSpin<QSpinBox>(computeGroup, 0, 8, 1);
    zBlockSpin_ = makeSpin<QSpinBox>(computeGroup, 1, 64, 1);
    connect(autoZChunkButton_, &QPushButton::clicked, this, &MainWindow::autoSelectZChunk);
    computeForm->addRow(QStringLiteral("Batch"), batchSpin_);
    computeForm->addRow(QStringLiteral("Z Chunk"), zChunkSpin_);
    computeForm->addRow(autoZChunkButton_);
    computeForm->addRow(QStringLiteral("GPU"), gpuSpin_);
    computeForm->addRow(QStringLiteral("Z Block Split"), zBlockSpin_);
    settingsLayout->addWidget(computeGroup);

    auto* previewGroup = new QGroupBox(QStringLiteral("Preview"), settingsHost);
    auto* previewForm = new QFormLayout(previewGroup);
    livePreviewCheck_ = new QCheckBox(QStringLiteral("Enable Live Preview"), previewGroup);
    liveSliceSpin_ = makeSpin<QSpinBox>(previewGroup, 0, 8192, 1);
    previewForm->addRow(livePreviewCheck_);
    previewForm->addRow(QStringLiteral("Slice Index"), liveSliceSpin_);
    settingsLayout->addWidget(previewGroup);
    settingsLayout->addStretch();
    scroll->setWidget(settingsHost);

    auto* rightPanel = new QWidget(central);
    auto* rightLayout = new QVBoxLayout(rightPanel);

    auto* imageRow = new QWidget(rightPanel);
    auto* imageRowLayout = new QHBoxLayout(imageRow);
    imageRowLayout->setContentsMargins(0, 0, 0, 0);

    auto* cameraPanel = new QWidget(imageRow);
    cameraPanel->setStyleSheet("background-color:#1e1e1e;border-radius:6px;");
    auto* cameraPanelLayout = new QVBoxLayout(cameraPanel);
    auto* cameraTitle = new QLabel(QStringLiteral("Kinetix Projection"), cameraPanel);
    cameraTitle->setAlignment(Qt::AlignCenter);
    cameraTitle->setStyleSheet("color:#4fa7ff;font-weight:bold;font-size:16px;margin-bottom:5px;");
    cameraPanelLayout->addWidget(cameraTitle);
    cameraPreviewView_ = new ZoomPanImageView(cameraPanel);
    cameraPreviewView_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    cameraPreviewView_->setMinimumSize(420, 320);
    cameraPanelLayout->addWidget(cameraPreviewView_, 1);
    auto* cameraHistogramTitle = new QLabel(QStringLiteral("Projection Contrast"), cameraPanel);
    cameraHistogramTitle->setStyleSheet("color:#fff;");
    cameraPanelLayout->addWidget(cameraHistogramTitle);
    cameraHistogramWidget_ = new ContrastHistogramWidget(cameraPanel);
    cameraHistogramWidget_->setMinimumHeight(120);
    cameraHistogramWidget_->setMaximumHeight(180);
    cameraPanelLayout->addWidget(cameraHistogramWidget_);
    imageRowLayout->addWidget(cameraPanel, 1);

    auto* reconPanel = new QWidget(imageRow);
    reconPanel->setStyleSheet("background-color:#1e1e1e;border-radius:6px;");
    auto* reconPanelLayout = new QVBoxLayout(reconPanel);
    auto* reconTitle = new QLabel(QStringLiteral("CT Reconstruction Slice"), reconPanel);
    reconTitle->setAlignment(Qt::AlignCenter);
    reconTitle->setStyleSheet("color:#4fa7ff;font-weight:bold;font-size:16px;margin-bottom:5px;");
    reconPanelLayout->addWidget(reconTitle);
    reconPreviewView_ = new ZoomPanImageView(reconPanel);
    reconPreviewView_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    reconPreviewView_->setMinimumSize(420, 320);
    reconPanelLayout->addWidget(reconPreviewView_, 1);

    auto* sliceLayout = new QHBoxLayout();
    auto* sliceTitleLabel = new QLabel(QStringLiteral("Slice (Z):"), reconPanel);
    sliceTitleLabel->setStyleSheet("color:#fff;");
    previewSliceSlider_ = new QSlider(Qt::Horizontal, reconPanel);
    previewSliceSlider_->setTracking(true);
    previewSliceValueLabel_ = new QLabel(QStringLiteral("0"), reconPanel);
    previewSliceValueLabel_->setFixedWidth(70);
    previewSliceValueLabel_->setAlignment(Qt::AlignCenter);
    previewSliceValueLabel_->setStyleSheet("font-weight:bold;color:#fff;background:#333;border-radius:4px;");
    sliceLayout->addWidget(sliceTitleLabel);
    sliceLayout->addWidget(previewSliceSlider_, 1);
    sliceLayout->addWidget(previewSliceValueLabel_);
    reconPanelLayout->addLayout(sliceLayout);

    auto* reconHistogramTitle = new QLabel(QStringLiteral("Slice Contrast"), reconPanel);
    reconHistogramTitle->setStyleSheet("color:#fff;");
    reconPanelLayout->addWidget(reconHistogramTitle);
    reconHistogramWidget_ = new ContrastHistogramWidget(reconPanel);
    reconHistogramWidget_->setMinimumHeight(120);
    reconHistogramWidget_->setMaximumHeight(180);
    reconPanelLayout->addWidget(reconHistogramWidget_);
    imageRowLayout->addWidget(reconPanel, 1);
    rightLayout->addWidget(imageRow, 1);

    auto* controlContainer = new QWidget(rightPanel);
    auto* controlLayout = new QVBoxLayout(controlContainer);
    logEdit_ = new QTextEdit(controlContainer);
    logEdit_->setReadOnly(true);
    logEdit_->setMaximumHeight(150);
    controlLayout->addWidget(logEdit_);
    progressBar_ = new QProgressBar(controlContainer);
    controlLayout->addWidget(progressBar_);
    auto* buttonLayout = new QHBoxLayout();
    startButton_ = new QPushButton(QStringLiteral("Start Recon"), controlContainer);
    stopButton_ = new QPushButton(QStringLiteral("Stop"), controlContainer);
    saveButton_ = new QPushButton(QStringLiteral("Save Slices"), controlContainer);
    stopButton_->setEnabled(false);
    saveButton_->setEnabled(false);
    buttonLayout->addWidget(startButton_);
    buttonLayout->addWidget(stopButton_);
    buttonLayout->addWidget(saveButton_);
    controlLayout->addLayout(buttonLayout);
    rightLayout->addWidget(controlContainer);
    mainLayout->addWidget(rightPanel, 1);
    mainLayout->addWidget(scroll);

    startCameraWorker();
    startMotorWorker();
    updateMotorUiEnabled();
    QTimer::singleShot(0, this, &MainWindow::requestMotorConnect);

    connect(cameraCaptureButton_, &QPushButton::clicked, this, &MainWindow::startCameraCapture);
    connect(cameraLiveButton_, &QPushButton::clicked, this, &MainWindow::startCameraLivePreview);
    connect(cameraStopButton_, &QPushButton::clicked, this, &MainWindow::stopCameraCapture);
    connect(startButton_, &QPushButton::clicked, this, &MainWindow::startReconstruction);
    connect(stopButton_, &QPushButton::clicked, this, &MainWindow::stopReconstruction);
    connect(saveButton_, &QPushButton::clicked, this, &MainWindow::saveSlices);
    connect(previewSliceSlider_, &QSlider::valueChanged, this, [this](int value) {
        syncPreviewSliceControls(value);
        requestPreviewSlice(value);
    });
    connect(liveSliceSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
        syncPreviewSliceControls(value);
        requestPreviewSlice(value);
    });
    connect(volHSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int depth) {
        setPreviewSliceRange(depth);
    });
    connect(cameraHistogramWidget_, &ContrastHistogramWidget::windowChanged, this, [this](double vmin, double vmax) {
        currentCameraPreviewWindow_.lower = vmin;
        currentCameraPreviewWindow_.upper = vmax;
        renderCameraPreviewWithCurrentWindow();
    });
    connect(reconHistogramWidget_, &ContrastHistogramWidget::windowChanged, this, [this](double vmin, double vmax) {
        currentReconPreviewWindow_.lower = vmin;
        currentReconPreviewWindow_.upper = vmax;
        renderPreviewWithCurrentWindow();
    });

    connect(&controller_, &ReconstructionController::logMessage, this, &MainWindow::appendLog);
    connect(&controller_, &ReconstructionController::progressUpdated, this, [this](int current, int total) {
        progressBar_->setMaximum(total);
        progressBar_->setValue(current);
    });
    connect(&controller_, &ReconstructionController::previewSliceReady, this, &MainWindow::updatePreviewSlice);
    connect(&controller_, &ReconstructionController::busyChanged, this, &MainWindow::setBusy);
    connect(&controller_, &ReconstructionController::reconstructionCompleted, this, [this](bool hasVolume) {
        appendLog(QStringLiteral("Reconstruction completed."));
        if (!hasVolume)
        {
            appendLog(QStringLiteral("Offline reconstruction saved slices during reconstruction; random volume browsing and Save button are disabled."));
        }
        saveButton_->setEnabled(hasVolume);
    });
    connect(&controller_, &ReconstructionController::reconstructionFailed, this, [this](const QString& error) {
        appendLog(QStringLiteral("Error: %1").arg(error));
    });
    connect(&controller_, &ReconstructionController::corComputed, this, [this](double shift) {
        corShiftSpin_->setValue(shift);
    });
    connect(&controller_, &ReconstructionController::volumeGeometryResolved, this, [this](int volW, int volH) {
        volWSpin_->setValue(volW);
        volHSpin_->setValue(volH);
        setPreviewSliceRange(volH);
    });
    connect(&controller_, &ReconstructionController::saveCompleted, this, [this]() {
        appendLog(QStringLiteral("Save completed."));
    });
    connect(&controller_, &ReconstructionController::saveFailed, this, [this](const QString& error) {
        appendLog(QStringLiteral("Save error: %1").arg(error));
    });

    const QString sessionConfigFile = sessionConfigPath();
    AppConfig loadedConfig = AppConfig::defaults();
    QString loadError;
    if (AppConfig::tryLoadFromJsonFile(sessionConfigFile, &loadedConfig, &loadError))
    {
        applyConfig(loadedConfig);
        appendLog(QStringLiteral("Loaded session parameters from last_session.json."));
    }
    else
    {
        applyDefaults();
        if (QFileInfo::exists(sessionConfigFile))
        {
            appendLog(QStringLiteral("Failed to load last_session.json: %1").arg(loadError));
        }
    }
    appendLog(QStringLiteral("App executable: %1").arg(QCoreApplication::applicationFilePath()));
    appendLog(QStringLiteral("Build stamp: %1 %2")
        .arg(QString::fromLatin1(__DATE__), QString::fromLatin1(__TIME__)));
}

MainWindow::~MainWindow()
{
    shutdownWorkers();
}

void MainWindow::applyDefaults()
{
    applyConfig(AppConfig::defaults());
}

void MainWindow::applyConfig(const AppConfig& config)
{
    sessionConfig_ = config;
    watchDirEdit_->setText(config.io.watchDir);
    saveDirEdit_->setText(config.io.saveDir);
    correctionModeCombo_->setCurrentText(toString(config.io.correctionMode));
    reconMethodCombo_->setCurrentText(toString(config.io.reconMethod));
    tomoPrefixEdit_->setText(config.io.tomoPrefix);
    flatPrefixEdit_->setText(config.io.flatPrefix);
    darkPrefixEdit_->setText(config.io.darkPrefix);
    paddingSpin_->setValue(config.io.inputPadding);
    extCombo_->setCurrentText(config.io.inputExt);
    file0Edit_->setText(config.io.filePattern0);
    file180Edit_->setText(config.io.filePattern180);
    savePrefixEdit_->setText(config.io.filePrefix);
    saveDepthCombo_->setCurrentText(toString(config.io.saveDepth));
    saveFormatCombo_->setCurrentText(config.io.saveFormat);
    tiffThreadsSpin_->setValue(config.io.tiffThreads);
    volWSpin_->setValue(config.geometry.volW);
    volHSpin_->setValue(config.geometry.volH);
    totalAnglesSpin_->setValue(config.geometry.totalAngles);
    corAutoCheck_->setChecked(config.geometry.corAuto);
    corShiftSpin_->setValue(config.geometry.corShift);
    angleOffsetSpin_->setValue(config.geometry.angleOffsetDeg);
    phaseEnableCheck_->setChecked(config.phase.enable);
    energySpin_->setValue(config.phase.energyKev);
    distanceSpin_->setValue(config.phase.distanceM);
    pixelSpin_->setValue(config.phase.pixelSizeUm);
    alphaSpin_->setValue(config.phase.alphaRatio);
    preprocessMethodCombo_->setCurrentText(toString(config.preprocess.method));
    airValueSpin_->setValue(config.preprocess.airValue);
    taperSpin_->setValue(config.preprocess.taperWidth);
    filterCombo_->setCurrentText(toString(config.preprocess.filterType));
    batchSpin_->setValue(config.compute.batchSize);
    zChunkSpin_->setValue(config.compute.zChunkSize);
    gpuSpin_->setValue(config.compute.gpuIndex);
    zBlockSpin_->setValue(config.compute.zBlockSplit);
    livePreviewCheck_->setChecked(config.preview.enableLive);
    setPreviewSliceRange(config.geometry.volH);
    syncPreviewSliceControls(config.preview.liveSliceIndex);
    updateReconMethodUiState();
}

AppConfig MainWindow::configFromUi() const
{
    AppConfig cfg = sessionConfig_;
    cfg.io.watchDir = watchDirEdit_->text();
    cfg.io.saveDir = saveDirEdit_->text();
    cfg.io.correctionMode = correctionModeFromString(correctionModeCombo_->currentText());
    cfg.io.reconMethod = reconMethodFromString(reconMethodCombo_->currentText());
    cfg.io.tomoPrefix = tomoPrefixEdit_->text();
    cfg.io.flatPrefix = flatPrefixEdit_->text();
    cfg.io.darkPrefix = darkPrefixEdit_->text();
    cfg.io.inputPadding = paddingSpin_->value();
    cfg.io.inputExt = extCombo_->currentText();
    cfg.io.filePattern0 = file0Edit_->text();
    cfg.io.filePattern180 = file180Edit_->text();
    cfg.io.filePrefix = savePrefixEdit_->text();
    cfg.io.saveDepth = saveDepthFromString(saveDepthCombo_->currentText());
    cfg.io.saveFormat = saveFormatCombo_->currentText();
    cfg.io.tiffThreads = tiffThreadsSpin_->value();
    cfg.geometry.volW = volWSpin_->value();
    cfg.geometry.volH = volHSpin_->value();
    cfg.geometry.totalAngles = totalAnglesSpin_->value();
    cfg.geometry.corAuto = corAutoCheck_->isChecked();
    cfg.geometry.corShift = static_cast<float>(corShiftSpin_->value());
    cfg.geometry.angleOffsetDeg = static_cast<float>(angleOffsetSpin_->value());
    cfg.phase.enable = phaseEnableCheck_->isChecked();
    if (cfg.io.reconMethod == ReconMethod::OfflineFBP
        || cfg.io.reconMethod == ReconMethod::OfflineGridrec)
    {
        cfg.phase.enable = false;
    }
    cfg.phase.energyKev = static_cast<float>(energySpin_->value());
    cfg.phase.distanceM = static_cast<float>(distanceSpin_->value());
    cfg.phase.pixelSizeUm = static_cast<float>(pixelSpin_->value());
    cfg.phase.alphaRatio = static_cast<float>(alphaSpin_->value());
    cfg.preprocess.method = preprocessMethodFromString(preprocessMethodCombo_->currentText());
    cfg.preprocess.airValue = static_cast<float>(airValueSpin_->value());
    cfg.preprocess.taperWidth = taperSpin_->value();
    cfg.preprocess.filterType = filterTypeFromString(filterCombo_->currentText());
    cfg.compute.batchSize = batchSpin_->value();
    cfg.compute.zChunkSize = zChunkSpin_->value();
    cfg.compute.gpuIndex = gpuSpin_->value();
    cfg.compute.zBlockSplit = zBlockSpin_->value();
    cfg.preview.enableLive = livePreviewCheck_->isChecked();
    cfg.preview.liveSliceIndex = liveSliceSpin_->value();
    return cfg;
}

bool MainWindow::saveSessionConfig()
{
    sessionConfig_ = configFromUi();
    const bool saved = sessionConfig_.saveToJsonFile(sessionConfigPath());
    if (!saved)
    {
        appendLog(QStringLiteral("Failed to save session parameters to last_session.json."));
    }
    return saved;
}

QString MainWindow::sessionConfigPath() const
{
    return QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("last_session.json"));
}

void MainWindow::setBusy(bool busy)
{
    startButton_->setEnabled(!busy);
    stopButton_->setEnabled(busy);
    saveButton_->setEnabled(!busy && controller_.hasVolume());
}

void MainWindow::appendLog(const QString& text)
{
    logEdit_->append(text);
}

void MainWindow::browseWatchDir()
{
    const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Input Directory"), watchDirEdit_->text());
    if (!path.isEmpty())
    {
        watchDirEdit_->setText(path);
    }
}

void MainWindow::browseSaveDir()
{
    const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Output Directory"), saveDirEdit_->text());
    if (!path.isEmpty())
    {
        saveDirEdit_->setText(path);
    }
}

void MainWindow::browseCameraSaveDir()
{
    const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Camera Output Directory"), cameraSaveDirEdit_->text());
    if (!path.isEmpty())
    {
        cameraSaveDirEdit_->setText(path);
    }
}

void MainWindow::startCameraWorker()
{
    cameraWorker_ = new PvcamCameraWorker();
    cameraWorker_->moveToThread(&cameraThread_);
    connect(cameraWorker_, &PvcamCameraWorker::cameraStateChanged, this, [this](const QString& state) {
        if (shuttingDown_) { return; }
        cameraStateLabel_->setText(state);
    });
    connect(cameraWorker_, &PvcamCameraWorker::frameReady, this, &MainWindow::updateCameraFrame);
    connect(cameraWorker_, &PvcamCameraWorker::captureStatsChanged, this, [this](qint64 captured, int queued, qint64 consumed, qint64 dropped) {
        if (shuttingDown_) { return; }
        cameraStatsLabel_->setText(QStringLiteral("cap %1 / q %2 / use %3 / drop %4")
            .arg(captured)
            .arg(queued)
            .arg(consumed)
            .arg(dropped));
    });
    connect(cameraWorker_, &PvcamCameraWorker::logMessage, this, &MainWindow::appendLog);
    connect(cameraWorker_, &PvcamCameraWorker::captureFinished, this, [this](bool ok, const QString& message) {
        if (shuttingDown_) { return; }
        cameraCaptureButton_->setEnabled(true);
        cameraLiveButton_->setEnabled(true);
        cameraStopButton_->setEnabled(false);
        appendLog(QStringLiteral("Kinetix capture %1: %2").arg(ok ? QStringLiteral("ok") : QStringLiteral("failed"), message));
    });
    cameraThread_.start();
}

void MainWindow::startCameraCapture()
{
    if (!cameraWorker_)
    {
        appendLog(QStringLiteral("Kinetix worker is unavailable."));
        return;
    }
    if (cameraTriggerModeCombo_->currentText().startsWith(QStringLiteral("External")))
    {
        appendLog(QStringLiteral("External trigger mode is reserved for the next integration step; using internal mode only now."));
        return;
    }

    lastRawCameraFrame_.release();
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;
    cameraHistogramWidget_->resetWindow();
    cameraPreviewFitInitialized_ = false;
    cameraPreviewView_->setImage(QPixmap());
    cameraPreviewView_->resetTransform();
    cameraCaptureButton_->setEnabled(false);
    cameraLiveButton_->setEnabled(false);
    cameraStopButton_->setEnabled(true);

    PvcamCaptureSettings settings;
    settings.frameCount = cameraFrameCountSpin_->value();
    settings.exposureMs = cameraExposureSpin_->value();
    settings.roiWidth = cameraRoiWidthSpin_->value();
    settings.roiHeight = cameraRoiHeightSpin_->value();
    settings.fullFrame = cameraFullFrameCheck_->isChecked();
    settings.queueCapacity = cameraQueueCapacitySpin_->value();
    settings.saveRaw = cameraSaveRawCheck_->isChecked();
    settings.outputDir = cameraSaveDirEdit_->text();

    appendLog(QStringLiteral("Starting Kinetix capture..."));
    QMetaObject::invokeMethod(
        cameraWorker_,
        "startCapture",
        Qt::QueuedConnection,
        Q_ARG(ct::PvcamCaptureSettings, settings));
}

void MainWindow::startCameraLivePreview()
{
    if (!cameraWorker_)
    {
        appendLog(QStringLiteral("Kinetix worker is unavailable."));
        return;
    }
    if (cameraTriggerModeCombo_->currentText().startsWith(QStringLiteral("External")))
    {
        appendLog(QStringLiteral("External trigger mode is reserved for the next integration step; using internal live preview now."));
        return;
    }

    lastRawCameraFrame_.release();
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;
    cameraHistogramWidget_->resetWindow();
    cameraPreviewFitInitialized_ = false;
    cameraPreviewView_->setImage(QPixmap());
    cameraPreviewView_->resetTransform();
    cameraCaptureButton_->setEnabled(false);
    cameraLiveButton_->setEnabled(false);
    cameraStopButton_->setEnabled(true);

    PvcamCaptureSettings settings;
    settings.frameCount = 1;
    settings.exposureMs = cameraExposureSpin_->value();
    settings.roiWidth = cameraRoiWidthSpin_->value();
    settings.roiHeight = cameraRoiHeightSpin_->value();
    settings.fullFrame = cameraFullFrameCheck_->isChecked();
    settings.queueCapacity = 1;
    settings.saveRaw = false;
    settings.previewOnly = true;
    settings.outputDir = cameraSaveDirEdit_->text();

    appendLog(QStringLiteral("Starting Kinetix live preview: exposure=%1 ms, ROI=%2x%3%4.")
        .arg(settings.exposureMs)
        .arg(settings.roiWidth)
        .arg(settings.roiHeight)
        .arg(settings.fullFrame ? QStringLiteral(", full frame") : QString()));
    QMetaObject::invokeMethod(
        cameraWorker_,
        "startCapture",
        Qt::QueuedConnection,
        Q_ARG(ct::PvcamCaptureSettings, settings));
}

void MainWindow::stopCameraCapture()
{
    if (!cameraWorker_)
    {
        return;
    }
    QMetaObject::invokeMethod(cameraWorker_, "stopCapture", Qt::DirectConnection);
}

void MainWindow::updateCameraFrame(const CameraFrame& frame)
{
    if (shuttingDown_ || frame.image.empty())
    {
        return;
    }

    ++cameraPreviewFrameCount_;
    frame.image.convertTo(lastRawCameraFrame_, CV_32F);

    const bool initializeWindow =
        cameraPreviewFrameCount_ == 1 || !cameraHistogramWidget_->isInitializedFromData();
    if (initializeWindow)
    {
        const QSignalBlocker blocker(cameraHistogramWidget_);
        cameraHistogramWidget_->updateData(lastRawCameraFrame_);
        currentCameraPreviewWindow_ = cameraHistogramWidget_->window();
    }

    if (cameraPreviewFrameCount_ == 1)
    {
        appendLog(QStringLiteral("Kinetix preview first frame: %1x%2, bitDepth=%3, cvType=%4, stride=%5.")
            .arg(frame.width)
            .arg(frame.height)
            .arg(frame.bitDepth)
            .arg(frame.cvType)
            .arg(frame.strideBytes));
    }

    const qint64 nowMs = frame.timestampMs > 0 ? frame.timestampMs : lastCameraPreviewRenderMs_ + 33;
    if (cameraPreviewFrameCount_ == 1 || nowMs - lastCameraPreviewRenderMs_ >= 33)
    {
        lastCameraPreviewRenderMs_ = nowMs;
        renderCameraPreviewWithCurrentWindow();
    }
}

void MainWindow::renderCameraPreviewWithCurrentWindow()
{
    if (lastRawCameraFrame_.empty())
    {
        return;
    }
    if (!currentCameraPreviewWindow_.isValid())
    {
        currentCameraPreviewWindow_ = PreviewSliceRenderer::estimateRobustWindow(lastRawCameraFrame_);
    }
    if (!currentCameraPreviewWindow_.isValid())
    {
        return;
    }

    const QImage image = PreviewSliceRenderer::renderFloatSlice(lastRawCameraFrame_, currentCameraPreviewWindow_);
    if (!image.isNull())
    {
        cameraPreviewView_->setImage(QPixmap::fromImage(image));
        if (!cameraPreviewFitInitialized_)
        {
            cameraPreviewView_->fitWindow();
            cameraPreviewFitInitialized_ = true;
        }
    }
}

void MainWindow::startReconstruction()
{
    saveSessionConfig();
    progressBar_->setValue(0);
    lastRawReconPreviewSlice_.release();
    reconHistogramWidget_->resetWindow();
    currentReconPreviewWindow_ = reconHistogramWidget_->window();
    reconPreviewFitInitialized_ = false;
    reconPreviewView_->setImage(QPixmap());
    reconPreviewView_->resetTransform();
    setPreviewSliceRange(volHSpin_->value());
    syncPreviewSliceControls(liveSliceSpin_->value());
    appendLog(QStringLiteral("Starting reconstruction..."));
    controller_.startReconstruction(configFromUi());
}

void MainWindow::stopReconstruction()
{
    controller_.stopReconstruction();
}

void MainWindow::saveSlices()
{
    SaveRequest request;
    request.config = controller_.lastConfig();
    request.config.io.saveDir = saveDirEdit_->text();
    request.config.io.filePrefix = savePrefixEdit_->text();
    request.config.io.saveDepth = saveDepthFromString(saveDepthCombo_->currentText());
    request.config.io.saveFormat = saveFormatCombo_->currentText();
    request.config.io.tiffThreads = tiffThreadsSpin_->value();
    request.volume = controller_.latestVolume();
    if (reconHistogramWidget_->hasWindow())
    {
        request.useDisplayWindow = true;
        request.displayMin = static_cast<float>(reconHistogramWidget_->window().lower);
        request.displayMax = static_cast<float>(reconHistogramWidget_->window().upper);
    }
    if (!request.volume)
    {
        appendLog(QStringLiteral("No reconstruction volume available."));
        return;
    }
    controller_.startSave(request);
}

void MainWindow::updatePreviewSlice(const cv::Mat& slice, bool updateHistogram)
{
    if (slice.empty())
    {
        return;
    }

    lastRawReconPreviewSlice_ = slice.clone();
    if (updateHistogram)
    {
        reconHistogramWidget_->updateData(lastRawReconPreviewSlice_);
        currentReconPreviewWindow_ = reconHistogramWidget_->window();
    }
    else if (!currentReconPreviewWindow_.isValid() && reconHistogramWidget_->hasWindow())
    {
        currentReconPreviewWindow_ = reconHistogramWidget_->window();
    }
    renderPreviewWithCurrentWindow();
}

void MainWindow::renderPreviewWithCurrentWindow()
{
    if (lastRawReconPreviewSlice_.empty() || !currentReconPreviewWindow_.isValid())
    {
        return;
    }

    const QImage image = PreviewSliceRenderer::renderFloatSlice(lastRawReconPreviewSlice_, currentReconPreviewWindow_);
    if (!image.isNull())
    {
        reconPreviewView_->setImage(QPixmap::fromImage(image));
        if (!reconPreviewFitInitialized_)
        {
            reconPreviewView_->fitWindow();
            reconPreviewFitInitialized_ = true;
        }
    }
}

void MainWindow::setPreviewSliceRange(int depth)
{
    const int maxSlice = std::max(0, depth - 1);
    previewSliceSlider_->setRange(0, maxSlice);
    liveSliceSpin_->setRange(0, maxSlice);
    syncPreviewSliceControls(std::clamp(liveSliceSpin_->value(), 0, maxSlice));
}

void MainWindow::syncPreviewSliceControls(int value)
{
    const int clampedValue = std::clamp(value, previewSliceSlider_->minimum(), previewSliceSlider_->maximum());
    {
        const QSignalBlocker blockSpin(liveSliceSpin_);
        liveSliceSpin_->setValue(clampedValue);
    }
    {
        const QSignalBlocker blockSlider(previewSliceSlider_);
        previewSliceSlider_->setValue(clampedValue);
    }
    updatePreviewSliceLabel(clampedValue);
}

void MainWindow::requestPreviewSlice(int value)
{
    controller_.setPreviewSliceIndex(value);
}

void MainWindow::updatePreviewSliceLabel(int value)
{
    previewSliceValueLabel_->setText(QString::number(value));
}

void MainWindow::updateReconMethodUiState()
{
    if (reconMethodCombo_ == nullptr)
    {
        return;
    }

    const ReconMethod method = reconMethodFromString(reconMethodCombo_->currentText());
    const bool offline = method == ReconMethod::OfflineFBP || method == ReconMethod::OfflineGridrec;

    if (batchSpin_ != nullptr)
    {
        batchSpin_->setEnabled(!offline);
        batchSpin_->setToolTip(offline
            ? QStringLiteral("Offline FBP/Gridrec reconstruct by full-angle sinogram z chunks. Batch is not a reconstruction parameter.")
            : QStringLiteral("Projection batch size for real-time FBP incremental reconstruction."));
    }
    if (zChunkSpin_ != nullptr)
    {
        zChunkSpin_->setEnabled(true);
        zChunkSpin_->setToolTip(offline
            ? QStringLiteral("Height chunk size for offline sinogram reconstruction.")
            : QStringLiteral("GPU slab height for real-time FBP."));
    }
    if (autoZChunkButton_ != nullptr)
    {
        autoZChunkButton_->setEnabled(true);
        autoZChunkButton_->setToolTip(QStringLiteral("Estimate Z Chunk from current reconstruction method and available GPU memory."));
    }

    if (phaseEnableCheck_ != nullptr)
    {
        if (offline && phaseEnableCheck_->isChecked())
        {
            const QSignalBlocker blocker(phaseEnableCheck_);
            phaseEnableCheck_->setChecked(false);
        }
        phaseEnableCheck_->setEnabled(!offline);
        phaseEnableCheck_->setToolTip(offline
            ? QStringLiteral("Paganin phase retrieval is disabled for offline FBP/Gridrec. Use camera-side preprocessing and point Watch Dir to the processed projection folder.")
            : QStringLiteral("Apply Paganin phase retrieval to each projection before real-time FBP."));
    }

    const bool phaseParameterEnabled = !offline;
    if (energySpin_ != nullptr) { energySpin_->setEnabled(phaseParameterEnabled); }
    if (distanceSpin_ != nullptr) { distanceSpin_->setEnabled(phaseParameterEnabled); }
    if (pixelSpin_ != nullptr) { pixelSpin_->setEnabled(phaseParameterEnabled); }
    if (alphaSpin_ != nullptr) { alphaSpin_->setEnabled(phaseParameterEnabled); }

    if (previewSliceSlider_ != nullptr)
    {
        previewSliceSlider_->setEnabled(!offline);
        previewSliceSlider_->setToolTip(offline
            ? QStringLiteral("Offline FBP/Gridrec previews the latest completed z chunk; random slice browsing is disabled.")
            : QStringLiteral("Select reconstruction slice."));
    }
    if (liveSliceSpin_ != nullptr)
    {
        liveSliceSpin_->setEnabled(!offline);
        liveSliceSpin_->setToolTip(offline
            ? QStringLiteral("Offline FBP/Gridrec previews chunk output directly; Slice Index is disabled.")
            : QStringLiteral("Live preview slice index."));
    }
}

void MainWindow::autoSelectZChunk()
{
    const int width = volWSpin_->value();
    const int height = volHSpin_->value();
    const int angles = totalAnglesSpin_->value();
    const int gpuIndex = gpuSpin_->value();
    const ReconMethod method = reconMethodFromString(reconMethodCombo_->currentText());

    if (width <= 0 || height <= 0 || angles <= 0)
    {
        appendLog(QStringLiteral("Auto Z Chunk failed: invalid geometry."));
        return;
    }
    if (cudaSetDevice(gpuIndex) != cudaSuccess)
    {
        appendLog(QStringLiteral("Auto Z Chunk failed: cannot select GPU %1.").arg(gpuIndex));
        return;
    }
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
    {
        appendLog(QStringLiteral("Auto Z Chunk failed: cudaMemGetInfo failed."));
        return;
    }

    const int paddedLength = [&]() {
        int padded = 1;
        while (padded < width * 2) { padded <<= 1; }
        return padded;
    }();
    const int freqWidth = paddedLength / 2 + 1;
    const double safety = 0.75;
    const double usableBytes = static_cast<double>(freeBytes) * safety;
    const double floatBytes = sizeof(float);
    const double complexBytes = sizeof(float) * 2.0;

    double bytesPerZ = 0.0;
    if (method == ReconMethod::OfflineGridrec)
    {
        bytesPerZ =
            static_cast<double>(angles) * width * floatBytes
            + static_cast<double>(angles) * paddedLength * floatBytes
            + static_cast<double>(angles) * freqWidth * complexBytes
            + static_cast<double>(width) * width * complexBytes * 2.0
            + static_cast<double>(width) * width * floatBytes;
    }
    else
    {
        bytesPerZ =
            static_cast<double>(angles) * width * floatBytes
            + static_cast<double>(angles) * paddedLength * floatBytes
            + static_cast<double>(angles) * freqWidth * complexBytes
            + static_cast<double>(width) * width * floatBytes;
    }

    if (bytesPerZ <= 0.0)
    {
        appendLog(QStringLiteral("Auto Z Chunk failed: invalid memory model."));
        return;
    }

    int recommended = static_cast<int>(std::floor(usableBytes / bytesPerZ));
    recommended = std::clamp(recommended, 1, std::min(height, zChunkSpin_->maximum()));
    if (recommended >= 16)
    {
        recommended = (recommended / 16) * 16;
    }
    zChunkSpin_->setValue(recommended);

    appendLog(QStringLiteral("Auto Z Chunk: method=%1 free=%2 GiB usable=%3 GiB -> Z Chunk=%4.")
        .arg(toString(method))
        .arg(static_cast<double>(freeBytes) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 2)
        .arg(usableBytes / (1024.0 * 1024.0 * 1024.0), 0, 'f', 2)
        .arg(recommended));
}

void MainWindow::startMotorWorker()
{
    kohzuController_ = new KohzuController();
    kohzuController_->moveToThread(&kohzuThread_);
    connect(kohzuController_, &KohzuController::connectionChanged, this, [this](bool connected) {
        if (shuttingDown_) { return; }
        motorConnected_ = connected;
        motorConnectionLabel_->setText(connected ? QStringLiteral("Connected") : QStringLiteral("Disconnected"));
        updateMotorUiEnabled();
    });
    connect(kohzuController_, &KohzuController::axisUpdated, this, &MainWindow::updateMotorAxisPosition);
    connect(kohzuController_, &KohzuController::axisActiveChanged, this, &MainWindow::updateMotorAxisActive);
    connect(kohzuController_, &KohzuController::logMessage, this, &MainWindow::appendLog);
    connect(kohzuController_, &KohzuController::commandFinished, this, [this](const QString& label, bool ok, const QString& response) {
        if (shuttingDown_) { return; }
        appendLog(QStringLiteral("%1 %2: %3").arg(label, ok ? QStringLiteral("ok") : QStringLiteral("failed"), response));
    });
    kohzuThread_.start();
    configureMotorAxes();
}

void MainWindow::configureMotorAxes()
{
    if (!kohzuController_)
    {
        return;
    }
    QMetaObject::invokeMethod(kohzuController_, "clearAxes", Qt::QueuedConnection);
    for (auto it = motorAxes_.begin(); it != motorAxes_.end(); ++it)
    {
        QMetaObject::invokeMethod(kohzuController_, "addAxis", Qt::QueuedConnection, Q_ARG(int, it->axis));
    }
}

void MainWindow::requestMotorConnect()
{
    if (!kohzuController_)
    {
        return;
    }
    motorConnectionLabel_->setText(QStringLiteral("Connecting"));
    QMetaObject::invokeMethod(
        kohzuController_,
        "connectToController",
        Qt::QueuedConnection,
        Q_ARG(QString, motorHostEdit_->text().trimmed()),
        Q_ARG(quint16, static_cast<quint16>(motorPortSpin_->value())));
}

void MainWindow::updateMotorAxisPosition(int axis, qint64 positionPulse, const QString& statusRaw)
{
    const QString name = motorAxisNamesByNumber_.value(axis);
    if (name.isEmpty() || !motorAxes_.contains(name))
    {
        return;
    }

    MotorAxisUi& axisUi = motorAxes_[name];
    const double userValue = static_cast<double>(positionPulse) / axisUi.pulsePerUnit;
    axisUi.positionLabel->setText(QStringLiteral("%1 %2").arg(userValue, 0, 'f', 4).arg(axisUi.unit));
    axisUi.positionLabel->setToolTip(QStringLiteral("%1 pulse\n%2").arg(positionPulse).arg(statusRaw));
}

void MainWindow::updateMotorAxisActive(int axis, bool active)
{
    const QString name = motorAxisNamesByNumber_.value(axis);
    if (name.isEmpty() || !motorAxes_.contains(name))
    {
        return;
    }

    MotorAxisUi& axisUi = motorAxes_[name];
    axisUi.active = active;
    axisUi.minusButton->setEnabled(motorConnected_ && !active);
    axisUi.plusButton->setEnabled(motorConnected_ && !active);
}

void MainWindow::jogMotorAxis(const QString& name, int direction)
{
    auto it = motorAxes_.find(name);
    if (it == motorAxes_.end())
    {
        return;
    }
    if (!motorConnected_)
    {
        appendLog(QStringLiteral("Kohzu is not connected."));
        return;
    }
    if (it->active)
    {
        appendLog(QStringLiteral("%1 is still moving or settling.").arg(name));
        return;
    }

    const double step = it->stepSpin->value();
    const qint64 pulse = static_cast<qint64>(std::llround(step * it->pulsePerUnit)) * direction;
    if (pulse == 0)
    {
        appendLog(QStringLiteral("%1 step is too small for pulse conversion.").arg(name));
        return;
    }

    const QString label = QStringLiteral("%1 jog %2 pulse (%3 %4)")
        .arg(name)
        .arg(pulse)
        .arg(step * direction, 0, 'f', 4)
        .arg(it->unit);
    it->active = true;
    it->minusButton->setEnabled(false);
    it->plusButton->setEnabled(false);
    QMetaObject::invokeMethod(
        kohzuController_,
        "jogRelativePulse",
        Qt::QueuedConnection,
        Q_ARG(int, it->axis),
        Q_ARG(qint64, pulse),
        Q_ARG(int, 0),
        Q_ARG(QString, label));
}

void MainWindow::stopMotorAxis(const QString& name)
{
    auto it = motorAxes_.find(name);
    if (it == motorAxes_.end())
    {
        return;
    }
    if (!motorConnected_)
    {
        appendLog(QStringLiteral("Kohzu is not connected."));
        return;
    }

    QMetaObject::invokeMethod(
        kohzuController_,
        "stopAxis",
        Qt::QueuedConnection,
        Q_ARG(int, it->axis),
        Q_ARG(QString, QStringLiteral("%1 stop").arg(name)));
}

void MainWindow::updateMotorUiEnabled()
{
    motorHostEdit_->setEnabled(!motorConnected_);
    motorPortSpin_->setEnabled(!motorConnected_);
    for (auto it = motorAxes_.begin(); it != motorAxes_.end(); ++it)
    {
        it->minusButton->setEnabled(motorConnected_ && !it->active);
        it->plusButton->setEnabled(motorConnected_ && !it->active);
        it->stopButton->setEnabled(motorConnected_);
    }
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    saveSessionConfig();
    shutdownWorkers();
    event->accept();
}

void MainWindow::shutdownWorkers()
{
    if (shuttingDown_)
    {
        return;
    }
    shuttingDown_ = true;

    controller_.shutdown();

    if (cameraWorker_)
    {
        disconnect(cameraWorker_, nullptr, this, nullptr);
        PvcamCameraWorker* worker = cameraWorker_;
        QThread* guiThread = thread();
        worker->shutdown();
        QMetaObject::invokeMethod(worker, [worker, guiThread]() {
            worker->moveToThread(guiThread);
        }, Qt::BlockingQueuedConnection);
        cameraWorker_ = nullptr;
        cameraThread_.quit();
        cameraThread_.wait(7000);
        delete worker;
    }
    else
    {
        cameraThread_.quit();
        cameraThread_.wait(7000);
    }

    if (kohzuController_)
    {
        disconnect(kohzuController_, nullptr, this, nullptr);
        KohzuController* controller = kohzuController_;
        QThread* guiThread = thread();
        QMetaObject::invokeMethod(controller, [controller]() {
            controller->shutdown();
        }, Qt::BlockingQueuedConnection);
        QMetaObject::invokeMethod(controller, [controller, guiThread]() {
            controller->moveToThread(guiThread);
        }, Qt::BlockingQueuedConnection);
        kohzuController_ = nullptr;
        kohzuThread_.quit();
        kohzuThread_.wait(1500);
        delete controller;
    }
    else
    {
        kohzuThread_.quit();
        kohzuThread_.wait(1500);
    }
}

} // namespace ct
