#include "ui/MainWindow.h"

#include "core/common/PreviewSliceRenderer.h"
#include "ui/ContrastHistogramWidget.h"
#include "ui/ZoomPanImageView.h"

#include <QAbstractSpinBox>
#include <QCloseEvent>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFile>
#include <QFileSystemWatcher>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMetaObject>
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
#include <QTextStream>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <cuda_runtime.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

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
    double softMin;
    double softMax;
    double defaultSpeed;
    double defaultAccel;
};

const MotorAxisDefinition kMotorAxes[] = {
    { "X90", "parallel sample trim", "XA04A-R102", 16, "CH8-2", "um", 20.0, 10.0, 500.0, -2000.0, 2000.0, 100.0, 500.0 },
    { "X0", "vertical sample trim", "XA04A-R102", 15, "CH8-1", "um", 20.0, 10.0, 500.0, -2000.0, 2000.0, 100.0, 500.0 },
    { "R", "CT rotation", "RA10A-W01", 17, "CH9-1", "deg", 20000.0, 0.01, 1.0, -360.0, 360.0, 1.0, 5.0 },
    { "U", "goniometer vertical beam", "SA10A-RM01", 1, "CH1-1", "deg", 11904.7619, 0.001, 0.1, -5.0, 5.0, 0.05, 0.2 },
    { "V", "goniometer parallel beam", "SA10A-RB01", 2, "CH1-2", "deg", 14925.3731, 0.001, 0.1, -5.0, 5.0, 0.05, 0.2 },
    { "Z", "sample height", "ZA16A-32F01", 18, "CH9-2", "um", 100.0, 1.0, 100.0, -10000.0, 10000.0, 20.0, 100.0 },
    { "Y", "large horizontal stage", "XA16F-L2101", 3, "CH2-1", "um", 0.2, 10.0, 1000.0, -150000.0, 150000.0, 500.0, 2000.0 },
    { "Det_OT2_Z", "Kinetix detector vertical lift", "ZA16A-32F01", 22, "CH11-2", "um", 100.0, 1.0, 100.0, -10000.0, 10000.0, 20.0, 100.0 },
    { "Det_OT2_Y", "Kinetix detector horizontal travel", "XA16F-L2301", 9, "CH5-1", "um", 2.0, 10.0, 1000.0, -50000.0, 50000.0, 500.0, 2000.0 },
};

QString motorDisplayUnit(const QString& unit)
{
    if (unit == QStringLiteral("um"))
    {
        return QString::fromUtf8("\xCE\xBCm");
    }
    if (unit == QStringLiteral("deg"))
    {
        return QString::fromUtf8("\xC2\xB0");
    }
    return unit;
}

QLabel* makeCenteredHeader(const QString& text, QWidget* parent)
{
    auto* label = new QLabel(text, parent);
    label->setAlignment(Qt::AlignCenter);
    return label;
}

void configureMotorSpinBox(QDoubleSpinBox* spin, int decimals, int width)
{
    spin->setDecimals(decimals);
    spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    spin->setAlignment(Qt::AlignRight);
    spin->setFixedWidth(width);
}

QPushButton* makeCompactMotorButton(const QString& text, QWidget* parent)
{
    auto* button = new QPushButton(text, parent);
    button->setFixedSize(24, 24);
    button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    return button;
}

PvcamExposureUnit cameraExposureUnitFromText(const QString& text)
{
    if (text == QStringLiteral("us"))
    {
        return PvcamExposureUnit::Microseconds;
    }
    if (text == QStringLiteral("s"))
    {
        return PvcamExposureUnit::Seconds;
    }
    return PvcamExposureUnit::Milliseconds;
}

QString cameraSnapPathForIndex(const QString& outputDir, const QString& filePrefix, int filePadding, int index)
{
    const QString prefix = filePrefix.isEmpty() ? QStringLiteral("kinetix_frame_") : filePrefix;
    const int padding = std::clamp(filePadding, 1, 12);
    QDir dir(outputDir);
    std::ostringstream name;
    name << prefix.toStdString() << std::setw(padding) << std::setfill('0') << std::max(0, index) << ".tiff";
    return dir.filePath(QString::fromStdString(name.str()));
}

QString numberedPath(const QString& dirPath, const QString& prefix, int padding, int index, const QString& ext)
{
    QDir dir(dirPath);
    return dir.filePath(QStringLiteral("%1%2.%3")
        .arg(prefix)
        .arg(std::max(0, index), std::clamp(padding, 1, 12), 10, QLatin1Char('0'))
        .arg(ext));
}

QJsonObject pvcamSettingsToJson(const PvcamCaptureSettings& settings)
{
    QJsonObject obj;
    obj["EXPOSURE_VALUE"] = settings.exposureValue;
    switch (settings.exposureUnit)
    {
    case PvcamExposureUnit::Microseconds: obj["EXPOSURE_UNIT"] = "us"; break;
    case PvcamExposureUnit::Seconds: obj["EXPOSURE_UNIT"] = "s"; break;
    case PvcamExposureUnit::Milliseconds:
    default: obj["EXPOSURE_UNIT"] = "ms"; break;
    }
    obj["ROI_X"] = settings.roiX;
    obj["ROI_Y"] = settings.roiY;
    obj["ROI_WIDTH"] = settings.roiWidth;
    obj["ROI_HEIGHT"] = settings.roiHeight;
    return obj;
}

bool writeJsonFileAtomic(const QString& path, const QJsonObject& object)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    const QString tmpPath = path + QStringLiteral(".tmp");
    QFile file(tmpPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    file.close();
    QFile::remove(path);
    return QFile::rename(tmpPath, path);
}

bool readJsonFile(const QString& path, QJsonObject* object, QString* errorMessage = nullptr)
{
    if (object == nullptr)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Output JSON pointer is null.");
        }
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Failed to open JSON: %1").arg(path);
        }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Invalid JSON %1: %2").arg(path, parseError.errorString());
        }
        return false;
    }
    *object = doc.object();
    return true;
}

QString sanitizeQueueName(QString value)
{
    value = value.trimmed();
    if (value.isEmpty())
    {
        return QStringLiteral("ct_job");
    }
    for (QChar& ch : value)
    {
        if (!ch.isLetterOrNumber() && ch != QLatin1Char('_') && ch != QLatin1Char('-'))
        {
            ch = QLatin1Char('_');
        }
    }
    return value.left(80);
}

QString queueFileName(int queueIndex, const QString& baseName)
{
    return QStringLiteral("%1_%2.json")
        .arg(std::max(0, queueIndex), 6, 10, QLatin1Char('0'))
        .arg(sanitizeQueueName(baseName));
}

int queueIndexFromFileName(const QString& fileName)
{
    const int underscore = fileName.indexOf(QLatin1Char('_'));
    if (underscore <= 0)
    {
        return std::numeric_limits<int>::max();
    }
    bool ok = false;
    const int index = fileName.left(underscore).toInt(&ok);
    return ok ? index : std::numeric_limits<int>::max();
}

QString queueBaseNameFromPath(const QString& path)
{
    QString fileName = QFileInfo(path).fileName();
    if (fileName.endsWith(QStringLiteral(".renaming")))
    {
        fileName.chop(QStringLiteral(".renaming").size());
    }
    if (fileName.endsWith(QStringLiteral(".reorder")))
    {
        fileName.chop(QStringLiteral(".reorder").size());
    }
    if (fileName.endsWith(QStringLiteral(".json")))
    {
        fileName.chop(QStringLiteral(".json").size());
    }
    QString base = fileName;
    const int underscore = base.indexOf(QLatin1Char('_'));
    bool numericPrefix = false;
    base.left(underscore).toInt(&numericPrefix);
    if (underscore >= 0 && numericPrefix)
    {
        base = base.mid(underscore + 1);
    }
    return sanitizeQueueName(base);
}

QString manifestPathForProjectionDir(const QString& projectionDir)
{
    return QDir(projectionDir).filePath(QStringLiteral("scan_manifest.json"));
}

QString projectionDirFromJob(const QJsonObject& job)
{
    QString projectionDir = job.value(QStringLiteral("PROJECTION_DIR")).toString();
    if (!projectionDir.isEmpty())
    {
        return projectionDir;
    }
    const QString manifestPath = job.value(QStringLiteral("MANIFEST_PATH")).toString();
    if (!manifestPath.isEmpty())
    {
        return QFileInfo(manifestPath).absolutePath();
    }
    return {};
}

QJsonObject appConfigToJobObject(const AppConfig& cfg)
{
    QJsonObject job;
    job["RECON_METHOD"] = toString(cfg.io.reconMethod);

    QJsonObject geometry;
    geometry["COR_SHIFT"] = cfg.geometry.corShift;
    geometry["COR_AUTO"] = cfg.geometry.corAuto;
    job["GEOMETRY"] = geometry;

    QJsonObject preprocess;
    preprocess["METHOD"] = toString(cfg.preprocess.method);
    preprocess["AVE"] = cfg.preprocess.airValue;
    preprocess["TAPER_WIDTH"] = cfg.preprocess.taperWidth;
    preprocess["FILTER_TYPE"] = toString(cfg.preprocess.filterType);
    job["PREPROCESS"] = preprocess;

    QJsonObject compute;
    compute["BATCH_SIZE"] = cfg.compute.batchSize;
    compute["Z_CHUNK_SIZE"] = cfg.compute.zChunkSize;
    compute["GPU_INDEX"] = cfg.compute.gpuIndex;
    compute["Z_BLOCK_SPLIT"] = cfg.compute.zBlockSplit;
    job["COMPUTE"] = compute;

    QJsonObject io;
    io["CORRECTION_MODE"] = toString(cfg.io.correctionMode);
    io["DARK_PREFIX"] = cfg.io.darkPrefix;
    io["FILE_PREFIX"] = cfg.io.filePrefix;
    io["SAVE_DEPTH"] = toString(cfg.io.saveDepth);
    io["SAVE_FMT"] = cfg.io.saveFormat;
    io["TIFF_THREADS"] = cfg.io.tiffThreads;
    job["IO"] = io;

    return job;
}

void applyJobObjectToConfig(const QJsonObject& job, AppConfig& cfg)
{
    const QJsonObject legacyRecon = job.value(QStringLiteral("RECON_CONFIG")).toObject();
    if (!legacyRecon.isEmpty())
    {
        cfg = AppConfig::fromJsonObject(legacyRecon);
    }

    if (job.contains(QStringLiteral("RECON_METHOD")))
    {
        cfg.io.reconMethod = reconMethodFromString(job.value(QStringLiteral("RECON_METHOD")).toString(toString(cfg.io.reconMethod)));
    }
    const QJsonObject geometry = job.value(QStringLiteral("GEOMETRY")).toObject();
    if (!geometry.isEmpty())
    {
        cfg.geometry.corShift = static_cast<float>(geometry.value(QStringLiteral("COR_SHIFT")).toDouble(cfg.geometry.corShift));
        cfg.geometry.corAuto = geometry.value(QStringLiteral("COR_AUTO")).toBool(cfg.geometry.corAuto);
    }
    const QJsonObject preprocess = job.value(QStringLiteral("PREPROCESS")).toObject();
    if (!preprocess.isEmpty())
    {
        cfg.preprocess.method = preprocessMethodFromString(preprocess.value(QStringLiteral("METHOD")).toString(toString(cfg.preprocess.method)));
        cfg.preprocess.airValue = static_cast<float>(preprocess.value(QStringLiteral("AVE")).toDouble(cfg.preprocess.airValue));
        cfg.preprocess.taperWidth = preprocess.value(QStringLiteral("TAPER_WIDTH")).toInt(cfg.preprocess.taperWidth);
        cfg.preprocess.filterType = filterTypeFromString(preprocess.value(QStringLiteral("FILTER_TYPE")).toString(toString(cfg.preprocess.filterType)));
    }
    const QJsonObject compute = job.value(QStringLiteral("COMPUTE")).toObject();
    if (!compute.isEmpty())
    {
        cfg.compute.batchSize = compute.value(QStringLiteral("BATCH_SIZE")).toInt(cfg.compute.batchSize);
        cfg.compute.zChunkSize = compute.value(QStringLiteral("Z_CHUNK_SIZE")).toInt(cfg.compute.zChunkSize);
        cfg.compute.gpuIndex = compute.value(QStringLiteral("GPU_INDEX")).toInt(cfg.compute.gpuIndex);
        cfg.compute.zBlockSplit = compute.value(QStringLiteral("Z_BLOCK_SPLIT")).toInt(cfg.compute.zBlockSplit);
    }
    const QJsonObject io = job.value(QStringLiteral("IO")).toObject();
    if (!io.isEmpty())
    {
        cfg.io.correctionMode = correctionModeFromString(io.value(QStringLiteral("CORRECTION_MODE")).toString(toString(cfg.io.correctionMode)));
        cfg.io.darkPrefix = io.value(QStringLiteral("DARK_PREFIX")).toString(cfg.io.darkPrefix);
        cfg.io.filePrefix = io.value(QStringLiteral("FILE_PREFIX")).toString(cfg.io.filePrefix);
        cfg.io.saveDepth = saveDepthFromString(io.value(QStringLiteral("SAVE_DEPTH")).toString(toString(cfg.io.saveDepth)));
        cfg.io.saveFormat = io.value(QStringLiteral("SAVE_FMT")).toString(cfg.io.saveFormat);
        cfg.io.tiffThreads = std::clamp(io.value(QStringLiteral("TIFF_THREADS")).toInt(cfg.io.tiffThreads), 1, 64);
    }
}

QString reconResultFileName(const QString& jobPath)
{
    return QFileInfo(jobPath).completeBaseName() + QStringLiteral("_recon_result.json");
}

QString numberedFileName(const QString& prefix, int padding, int index, const QString& ext)
{
    return QStringLiteral("%1%2.%3")
        .arg(prefix)
        .arg(std::max(0, index), std::clamp(padding, 1, 12), 10, QLatin1Char('0'))
        .arg(ext);
}

void moveJobToFailed(const QString& jobPath, QJsonObject job, const QString& resultPath,
    const QString& stage, const QString& code, const QString& message, const QString& queueRoot)
{
    if (!job.contains(QStringLiteral("TYPE")))
    {
        job["TYPE"] = "CT_STREAM_RECON_JOB";
    }
    job["STATUS"] = "failed";
    job["JOB_PATH"] = jobPath;
    job["FINISHED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    job["FAILED_STAGE"] = stage;
    job["ERROR_CODE"] = code;
    job["ERROR_MESSAGE"] = message;
    job["RESULT_PATH"] = resultPath;
    writeJsonFileAtomic(jobPath, job);

    const QString failedDir = QDir(queueRoot).filePath(QStringLiteral("failed"));
    QDir().mkpath(failedDir);
    const QString targetPath = QDir(failedDir).filePath(QFileInfo(jobPath).fileName());
    QFile::remove(targetPath);
    QFile::rename(jobPath, targetPath);
}

qint64 previewIntervalForRoi(int width, int height)
{
    const qint64 pixels = static_cast<qint64>(std::max(1, width)) * static_cast<qint64>(std::max(1, height));
    if (pixels >= 2500LL * 2500LL)
    {
        return 200; // 5 FPS for full-frame Kinetix preview.
    }
    if (pixels >= 1500LL * 1500LL)
    {
        return 100; // 10 FPS for medium-large ROI.
    }
    return 33; // About 30 FPS for small ROI.
}

cv::Mat makeCameraPreviewMat(const cv::Mat& source)
{
    constexpr int kMaxPreviewSide = 1024;
    if (source.empty())
    {
        return {};
    }
    const int maxSide = std::max(source.cols, source.rows);
    if (maxSide <= kMaxPreviewSide)
    {
        return source.clone();
    }
    const double scale = static_cast<double>(kMaxPreviewSide) / static_cast<double>(maxSide);
    cv::Mat preview;
    cv::resize(source, preview, cv::Size(), scale, scale, cv::INTER_AREA);
    return preview;
}

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("C++ CT Workstation v1"));
    resize(1850, 980);
    cameraPreviewFitInitialized_ = false;
    reconPreviewFitInitialized_ = false;
    qRegisterMetaType<ct::CtFrameSaveItem>("ct::CtFrameSaveItem");

    auto* central = new QWidget(this);
    auto* mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(8, 8, 8, 8);
    mainLayout->setSpacing(8);
    setCentralWidget(central);

    auto* hardwareScroll = new QScrollArea(central);
    hardwareScroll->setWidgetResizable(true);
    hardwareScroll->setFixedWidth(470);
    hardwareScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
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
    cameraExposureSpin_ = makeSpin<QDoubleSpinBox>(cameraGroup, 0.001, 600000.0, 1.0);
    cameraExposureSpin_->setValue(10.0);
    cameraExposureSpin_->setDecimals(3);
    cameraExposureUnitCombo_ = new QComboBox(cameraGroup);
    cameraExposureUnitCombo_->addItems({ "us", "ms", "s" });
    cameraExposureUnitCombo_->setCurrentText(QStringLiteral("ms"));
    auto* cameraExposureLayout = new QHBoxLayout();
    cameraExposureLayout->setContentsMargins(0, 0, 0, 0);
    cameraExposureLayout->addWidget(cameraExposureSpin_, 1);
    cameraExposureLayout->addWidget(cameraExposureUnitCombo_);
    auto* cameraExposureWidget = new QWidget(cameraGroup);
    cameraExposureWidget->setLayout(cameraExposureLayout);
    cameraRoiXSpin_ = makeSpin<QSpinBox>(cameraGroup, 0, 3199, 1);
    cameraRoiYSpin_ = makeSpin<QSpinBox>(cameraGroup, 0, 3199, 1);
    cameraRoiWidthSpin_ = makeSpin<QSpinBox>(cameraGroup, 1, 3200, 1);
    cameraRoiHeightSpin_ = makeSpin<QSpinBox>(cameraGroup, 1, 3200, 1);
    cameraRoiXSpin_->setValue(0);
    cameraRoiYSpin_->setValue(0);
    cameraRoiWidthSpin_->setValue(3200);
    cameraRoiHeightSpin_->setValue(3200);
    cameraApplyRoiButton_ = new QPushButton(QStringLiteral("Apply ROI"), cameraGroup);
    cameraResetRoiButton_ = new QPushButton(QStringLiteral("Reset Full"), cameraGroup);
    auto* cameraRoiButtonLayout = new QHBoxLayout();
    cameraRoiButtonLayout->setContentsMargins(0, 0, 0, 0);
    cameraRoiButtonLayout->addWidget(cameraApplyRoiButton_);
    cameraRoiButtonLayout->addWidget(cameraResetRoiButton_);
    auto* cameraRoiButtonWidget = new QWidget(cameraGroup);
    cameraRoiButtonWidget->setLayout(cameraRoiButtonLayout);
    cameraSnapPrefixEdit_ = new QLineEdit(QStringLiteral("kinetix_frame_"), cameraGroup);
    cameraSnapPaddingSpin_ = makeSpin<QSpinBox>(cameraGroup, 1, 12, 1);
    cameraSnapPaddingSpin_->setValue(6);
    cameraSnapIndexSpin_ = makeSpin<QSpinBox>(cameraGroup, 0, 999999999, 1);
    cameraSnapIndexSpin_->setValue(0);
    cameraSaveDirEdit_ = new QLineEdit(QStringLiteral("E:/2026_Chun/2026_06_09/pvcam_capture_test"), cameraGroup);
    cameraSaveDirEdit_->setMinimumWidth(120);
    auto* cameraSaveBrowse = new QPushButton(QStringLiteral("..."), cameraGroup);
    connect(cameraSaveBrowse, &QPushButton::clicked, this, &MainWindow::browseCameraSaveDir);
    auto* cameraSaveLayout = new QHBoxLayout();
    cameraSaveLayout->addWidget(cameraSaveDirEdit_);
    cameraSaveLayout->addWidget(cameraSaveBrowse);
    auto* cameraSaveWidget = new QWidget(cameraGroup);
    cameraSaveWidget->setLayout(cameraSaveLayout);
    cameraSnapButton_ = new QPushButton(QStringLiteral("Snap"), cameraGroup);
    cameraLiveButton_ = new QPushButton(QStringLiteral("Open Live"), cameraGroup);
    auto* cameraButtonLayout = new QHBoxLayout();
    cameraButtonLayout->setContentsMargins(0, 0, 0, 0);
    cameraButtonLayout->addWidget(cameraSnapButton_);
    cameraButtonLayout->addWidget(cameraLiveButton_);
    auto* cameraButtonWidget = new QWidget(cameraGroup);
    cameraButtonWidget->setLayout(cameraButtonLayout);
    cameraForm->addRow(QStringLiteral("State"), cameraStateLabel_);
    cameraForm->addRow(QStringLiteral("Exposure"), cameraExposureWidget);
    cameraForm->addRow(QStringLiteral("ROI X"), cameraRoiXSpin_);
    cameraForm->addRow(QStringLiteral("ROI Y"), cameraRoiYSpin_);
    cameraForm->addRow(QStringLiteral("Width"), cameraRoiWidthSpin_);
    cameraForm->addRow(QStringLiteral("Height"), cameraRoiHeightSpin_);
    cameraForm->addRow(cameraRoiButtonWidget);
    cameraForm->addRow(QStringLiteral("Snap Prefix"), cameraSnapPrefixEdit_);
    cameraForm->addRow(QStringLiteral("Padding"), cameraSnapPaddingSpin_);
    cameraForm->addRow(QStringLiteral("Snap Index"), cameraSnapIndexSpin_);
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
    motorHostEdit_->hide();
    motorPortSpin_->hide();
    motorConnectionLabel_ = new QLabel(QStringLiteral("Disconnected"), motorConnectGroup);
    motorConnectionLabel_->hide();
    motorConnectGroup->hide();

    auto* ctGroup = new QGroupBox(QStringLiteral("CT Acquisition"), hardwareHost);
    auto* ctForm = new QFormLayout(ctGroup);
    ctForm->setContentsMargins(8, 8, 8, 8);
    ctForm->setHorizontalSpacing(6);
    ctForm->setVerticalSpacing(4);
    ctBeginningAngleSpin_ = makeSpin<QDoubleSpinBox>(ctGroup, -360.0, 360.0, 0.1);
    ctBeginningAngleSpin_->setDecimals(4);
    ctBeginningAngleSpin_->setValue(0.0);
    ctSavePathEdit_ = new QLineEdit(ctGroup);
    ctSavePathEdit_->setMinimumWidth(120);
    auto* ctSaveBrowse = new QPushButton(QStringLiteral("..."), ctGroup);
    connect(ctSaveBrowse, &QPushButton::clicked, this, &MainWindow::browseCtSavePath);
    auto* ctSaveLayout = new QHBoxLayout();
    ctSaveLayout->setContentsMargins(0, 0, 0, 0);
    ctSaveLayout->addWidget(ctSavePathEdit_);
    ctSaveLayout->addWidget(ctSaveBrowse);
    auto* ctSaveWidget = new QWidget(ctGroup);
    ctSaveWidget->setLayout(ctSaveLayout);
    ctProjectionNumSpin_ = makeSpin<QSpinBox>(ctGroup, 1, 20000, 1);
    ctProjectionNumSpin_->setValue(1080);
    ctFlatGapSpin_ = makeSpin<QSpinBox>(ctGroup, 0, 20000, 1);
    ctFlatGapSpin_->setValue(0);
    ctFlatMoveSpin_ = makeSpin<QDoubleSpinBox>(ctGroup, -500000.0, 500000.0, 100.0);
    ctFlatMoveSpin_->setDecimals(4);
    ctFlatMoveSpin_->setValue(0.0);
    ctFlatNumSpin_ = makeSpin<QSpinBox>(ctGroup, 0, 10000, 1);
    ctFlatNumSpin_->setValue(10);
    ctAutoCorCheck_ = new QCheckBox(QStringLiteral("Auto COR reference"), ctGroup);
    ctTiffThreadsSpin_ = makeSpin<QSpinBox>(ctGroup, 1, 64, 1);
    ctTiffThreadsSpin_->setValue(16);
    ctStatusLabel_ = new QLabel(QStringLiteral("Idle"), ctGroup);
    ctQueueLabel_ = new QLabel(QStringLiteral("Queue 0 / Saved 0"), ctGroup);
    ctStartButton_ = new QPushButton(QStringLiteral("Start CT"), ctGroup);
    ctStopButton_ = new QPushButton(QStringLiteral("Stop CT"), ctGroup);
    ctStopButton_->setEnabled(false);
    ctStopButton_->setStyleSheet(QStringLiteral("QPushButton { background: #b00020; color: white; font-weight: 600; }"));
    auto* ctButtonLayout = new QHBoxLayout();
    ctButtonLayout->setContentsMargins(0, 0, 0, 0);
    ctButtonLayout->addWidget(ctStartButton_);
    ctButtonLayout->addWidget(ctStopButton_);
    auto* ctButtonWidget = new QWidget(ctGroup);
    ctButtonWidget->setLayout(ctButtonLayout);
    ctForm->addRow(QStringLiteral("Beginning Angle"), ctBeginningAngleSpin_);
    ctForm->addRow(QStringLiteral("Save Path"), ctSaveWidget);
    ctForm->addRow(QStringLiteral("Projection Num"), ctProjectionNumSpin_);
    ctForm->addRow(QStringLiteral("Flat Gap"), ctFlatGapSpin_);
    ctForm->addRow(QStringLiteral("Flat Move um"), ctFlatMoveSpin_);
    ctForm->addRow(QStringLiteral("Flat Num"), ctFlatNumSpin_);
    ctForm->addRow(ctAutoCorCheck_);
    ctForm->addRow(QStringLiteral("CT TIFF Threads"), ctTiffThreadsSpin_);
    ctForm->addRow(QStringLiteral("Status"), ctStatusLabel_);
    ctForm->addRow(QStringLiteral("Save Queue"), ctQueueLabel_);
    ctForm->addRow(ctButtonWidget);
    hardwareLayout->addWidget(ctGroup);

    auto* axisGroup = new QGroupBox(QStringLiteral("Motor Axes"), hardwareHost);
    auto* axisLayout = new QGridLayout(axisGroup);
    axisLayout->setContentsMargins(8, 8, 8, 8);
    axisLayout->setHorizontalSpacing(4);
    axisLayout->setVerticalSpacing(4);
    axisLayout->setColumnMinimumWidth(0, 70);
    axisLayout->setColumnMinimumWidth(1, 78);
    axisLayout->setColumnMinimumWidth(2, 34);
    axisLayout->setColumnMinimumWidth(3, 126);
    axisLayout->setColumnMinimumWidth(4, 76);
    axisLayout->setColumnMinimumWidth(5, 40);
    motorMotionHeader_ = new QWidget(axisGroup);
    auto* motionHeaderLayout = new QGridLayout(motorMotionHeader_);
    motionHeaderLayout->setContentsMargins(0, 0, 0, 0);
    motionHeaderLayout->setHorizontalSpacing(4);
    motionHeaderLayout->setColumnMinimumWidth(0, 70);
    motionHeaderLayout->setColumnMinimumWidth(1, 78);
    motionHeaderLayout->setColumnMinimumWidth(2, 34);
    motionHeaderLayout->setColumnMinimumWidth(3, 126);
    motionHeaderLayout->setColumnMinimumWidth(4, 76);
    motionHeaderLayout->setColumnMinimumWidth(5, 40);
    motionHeaderLayout->addWidget(new QLabel(QString(), motorMotionHeader_), 0, 0);
    motionHeaderLayout->addWidget(makeCenteredHeader(QStringLiteral("Current"), motorMotionHeader_), 0, 1);
    motionHeaderLayout->addWidget(new QLabel(QString(), motorMotionHeader_), 0, 2);
    motionHeaderLayout->addWidget(makeCenteredHeader(QStringLiteral("Relative"), motorMotionHeader_), 0, 3);
    motionHeaderLayout->addWidget(makeCenteredHeader(QStringLiteral("Absolute"), motorMotionHeader_), 0, 4);
    motionHeaderLayout->addWidget(new QLabel(QString(), motorMotionHeader_), 0, 5);
    axisLayout->addWidget(motorMotionHeader_, 0, 0, 1, 6);

    motorSpeedHeader_ = new QWidget(axisGroup);
    auto* speedHeaderLayout = new QGridLayout(motorSpeedHeader_);
    speedHeaderLayout->setContentsMargins(0, 0, 0, 0);
    speedHeaderLayout->setHorizontalSpacing(4);
    speedHeaderLayout->setColumnMinimumWidth(0, 70);
    speedHeaderLayout->setColumnMinimumWidth(1, 78);
    speedHeaderLayout->setColumnMinimumWidth(2, 34);
    speedHeaderLayout->setColumnMinimumWidth(3, 126);
    speedHeaderLayout->setColumnMinimumWidth(4, 76);
    speedHeaderLayout->setColumnMinimumWidth(5, 40);
    speedHeaderLayout->addWidget(new QLabel(QString(), motorSpeedHeader_), 0, 0);
    speedHeaderLayout->addWidget(new QLabel(QString(), motorSpeedHeader_), 0, 1);
    speedHeaderLayout->addWidget(new QLabel(QString(), motorSpeedHeader_), 0, 2);
    speedHeaderLayout->addWidget(makeCenteredHeader(QStringLiteral("Speed (/s)"), motorSpeedHeader_), 0, 3);
    speedHeaderLayout->addWidget(makeCenteredHeader(QStringLiteral("Accel (/s^2)"), motorSpeedHeader_), 0, 4);
    axisLayout->addWidget(motorSpeedHeader_, 0, 0, 1, 6);
    motorSpeedHeader_->hide();

    int axisRow = 1;
    for (const MotorAxisDefinition& def : kMotorAxes)
    {
        MotorAxisUi axisUi;
        axisUi.name = QString::fromLatin1(def.name);
        axisUi.axis = def.axis;
        axisUi.pulsePerUnit = def.pulsePerUnit;
        axisUi.unit = motorDisplayUnit(QString::fromLatin1(def.unit));
        axisUi.softMin = def.softMin;
        axisUi.softMax = def.softMax;

        auto* axisName = new QLabel(axisUi.name, axisGroup);
        axisName->setToolTip(QStringLiteral("A%1 %2\n%3\n%4\nSoft range: %5 to %6 %7")
            .arg(def.axis)
            .arg(QString::fromLatin1(def.channel))
            .arg(QString::fromLatin1(def.model))
            .arg(QString::fromLatin1(def.function))
            .arg(def.softMin, 0, 'f', 4)
            .arg(def.softMax, 0, 'f', 4)
            .arg(axisUi.unit));
        axisUi.nameLabel = axisName;
        axisUi.positionLabel = new QLabel(QStringLiteral("-"), axisGroup);
        axisUi.positionLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        axisUi.positionLabel->setFixedWidth(78);
        axisUi.unitLabel = new QLabel(axisUi.unit, axisGroup);
        axisUi.unitLabel->setAlignment(Qt::AlignCenter);
        axisUi.unitLabel->setFixedWidth(34);
        axisUi.statusLabel = new QLabel(QStringLiteral(""), axisGroup);
        axisUi.statusLabel->setVisible(false);
        axisUi.relativeMinusButton = makeCompactMotorButton(QStringLiteral("-"), axisGroup);
        axisUi.relativeSpin = makeSpin<QDoubleSpinBox>(axisGroup, 0.0, std::abs(def.softMax - def.softMin), def.defaultStep);
        configureMotorSpinBox(axisUi.relativeSpin, 4, 76);
        axisUi.relativeSpin->setValue(def.defaultStep);
        axisUi.relativePlusButton = makeCompactMotorButton(QStringLiteral("+"), axisGroup);
        axisUi.absoluteSpin = makeSpin<QDoubleSpinBox>(axisGroup, def.softMin, def.softMax, def.defaultStep);
        configureMotorSpinBox(axisUi.absoluteSpin, 4, 76);
        axisUi.absoluteSpin->setValue(0.0);
        axisUi.goButton = new QPushButton(QStringLiteral("Go"), axisGroup);
        axisUi.goButton->setFixedWidth(40);
        axisUi.speedUnitLabel = new QLabel(axisUi.unit, axisGroup);
        axisUi.speedUnitLabel->setAlignment(Qt::AlignCenter);
        axisUi.speedUnitLabel->setFixedWidth(34);
        axisUi.speedSpin = makeSpin<QDoubleSpinBox>(axisGroup, 0.0, 1000000.0, def.defaultSpeed);
        configureMotorSpinBox(axisUi.speedSpin, 4, 76);
        axisUi.speedSpin->setValue(def.defaultSpeed);
        axisUi.accelSpin = makeSpin<QDoubleSpinBox>(axisGroup, 0.0, 1000000.0, def.defaultAccel);
        configureMotorSpinBox(axisUi.accelSpin, 4, 76);
        axisUi.accelSpin->setValue(def.defaultAccel);
        axisUi.speedUnitLabel->hide();
        axisUi.speedSpin->hide();
        axisUi.accelSpin->hide();

        axisLayout->addWidget(axisName, axisRow, 0);
        axisLayout->addWidget(axisUi.positionLabel, axisRow, 1);
        axisLayout->addWidget(axisUi.unitLabel, axisRow, 2);
        auto* relativeLayout = new QHBoxLayout();
        relativeLayout->setContentsMargins(0, 0, 0, 0);
        relativeLayout->setSpacing(0);
        relativeLayout->addWidget(axisUi.relativeMinusButton);
        relativeLayout->addWidget(axisUi.relativeSpin);
        relativeLayout->addWidget(axisUi.relativePlusButton);
        auto* relativeWidget = new QWidget(axisGroup);
        relativeWidget->setLayout(relativeLayout);
        relativeWidget->setFixedWidth(124);
        axisUi.relativeWidget = relativeWidget;
        axisLayout->addWidget(relativeWidget, axisRow, 3);
        axisLayout->addWidget(axisUi.absoluteSpin, axisRow, 4);
        axisLayout->addWidget(axisUi.goButton, axisRow, 5);
        axisLayout->addWidget(axisUi.speedUnitLabel, axisRow, 2);
        axisLayout->addWidget(axisUi.speedSpin, axisRow, 3);
        axisLayout->addWidget(axisUi.accelSpin, axisRow, 4);

        const QString axisKey = axisUi.name;
        connect(axisUi.relativeMinusButton, &QPushButton::clicked, this, [this, axisKey]() {
            moveMotorAxisRelative(axisKey, -1);
        });
        connect(axisUi.relativePlusButton, &QPushButton::clicked, this, [this, axisKey]() {
            moveMotorAxisRelative(axisKey, 1);
        });
        connect(axisUi.goButton, &QPushButton::clicked, this, [this, axisKey]() {
            moveMotorAxis(axisKey);
        });

        motorAxes_.insert(axisKey, axisUi);
        motorAxisNamesByNumber_.insert(axisUi.axis, axisKey);
        ++axisRow;
    }
    motorModeButton_ = new QPushButton(QStringLiteral("Speed"), axisGroup);
    motorStopAllButton_ = new QPushButton(QStringLiteral("Stop"), axisGroup);
    motorStopAllButton_->setStyleSheet(QStringLiteral("QPushButton { background: #b00020; color: white; font-weight: 600; }"));
    connect(motorModeButton_, &QPushButton::clicked, this, &MainWindow::toggleMotorSpeedMode);
    connect(motorStopAllButton_, &QPushButton::clicked, this, &MainWindow::stopAllMotorAxes);
    axisLayout->addWidget(motorModeButton_, axisRow, 0, 1, 3);
    axisLayout->addWidget(motorStopAllButton_, axisRow, 3, 1, 3);
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

    auto* reconQueueGroup = new QGroupBox(QStringLiteral("Recon Queue"), settingsHost);
    auto* reconQueueLayout = new QVBoxLayout(reconQueueGroup);
    reconQueueLayout->setContentsMargins(8, 8, 8, 8);
    reconQueueList_ = new QListWidget(reconQueueGroup);
    reconQueueList_->setMinimumHeight(80);
    reconQueueList_->setMaximumHeight(140);
    reconQueueLayout->addWidget(reconQueueList_);
    auto* reconQueueButtonLayout = new QHBoxLayout();
    reconQueueUpButton_ = new QPushButton(QStringLiteral("Up"), reconQueueGroup);
    reconQueueDownButton_ = new QPushButton(QStringLiteral("Down"), reconQueueGroup);
    reconQueueRemoveButton_ = new QPushButton(QStringLiteral("Remove"), reconQueueGroup);
    reconQueueButtonLayout->addWidget(reconQueueUpButton_);
    reconQueueButtonLayout->addWidget(reconQueueDownButton_);
    reconQueueButtonLayout->addWidget(reconQueueRemoveButton_);
    reconQueueLayout->addLayout(reconQueueButtonLayout);
    settingsLayout->addWidget(reconQueueGroup);

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

    connect(cameraSnapButton_, &QPushButton::clicked, this, &MainWindow::snapCameraFrame);
    connect(cameraLiveButton_, &QPushButton::clicked, this, &MainWindow::toggleCameraLivePreview);
    connect(cameraApplyRoiButton_, &QPushButton::clicked, this, &MainWindow::applyCameraRoi);
    connect(cameraResetRoiButton_, &QPushButton::clicked, this, &MainWindow::resetCameraRoiFullFrame);
    connect(ctStartButton_, &QPushButton::clicked, this, &MainWindow::startCtAcquisition);
    connect(ctStopButton_, &QPushButton::clicked, this, &MainWindow::stopCtAcquisition);
    connect(reconQueueUpButton_, &QPushButton::clicked, this, [this]() { moveSelectedReconQueueJob(-1); });
    connect(reconQueueDownButton_, &QPushButton::clicked, this, [this]() { moveSelectedReconQueueJob(1); });
    connect(reconQueueRemoveButton_, &QPushButton::clicked, this, &MainWindow::removeSelectedReconQueueJob);
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
        finishQueuedReconstruction(true, QStringLiteral("Reconstruction completed."));
    });
    connect(&controller_, &ReconstructionController::reconstructionFailed, this, [this](const QString& error) {
        appendLog(QStringLiteral("Error: %1").arg(error));
        finishQueuedReconstruction(false, error);
    });
    connect(&controller_, &ReconstructionController::corComputed, this, [this](double shift) {
        corShiftSpin_->setValue(shift);
        updateActiveQueueCor(shift);
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
    ensureReconQueueDirs();
    recoverStaleRunningReconJobs();
    reconQueueWatcher_ = new QFileSystemWatcher(this);
    const QString queueRoot = reconQueueRoot();
    const QStringList watchDirs = {
        QDir(queueRoot).filePath(QStringLiteral("pending")),
        QDir(queueRoot).filePath(QStringLiteral("running")),
        QDir(queueRoot).filePath(QStringLiteral("done")),
        QDir(queueRoot).filePath(QStringLiteral("failed")),
    };
    reconQueueWatcher_->addPaths(watchDirs);
    connect(reconQueueWatcher_, &QFileSystemWatcher::directoryChanged, this, [this](const QString&) {
        scheduleReconQueueRefresh();
    });
    reconQueueRefreshTimer_ = new QTimer(this);
    reconQueueRefreshTimer_->setSingleShot(true);
    reconQueueRefreshTimer_->setInterval(300);
    connect(reconQueueRefreshTimer_, &QTimer::timeout, this, &MainWindow::refreshReconQueue);
    reconQueuePollTimer_ = new QTimer(this);
    reconQueuePollTimer_->setInterval(3000);
    connect(reconQueuePollTimer_, &QTimer::timeout, this, [this]() {
        refreshReconQueue();
        startNextQueuedReconstruction();
    });
    reconQueuePollTimer_->start();
    refreshReconQueue();
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
    QTimer::singleShot(0, this, &MainWindow::startNextQueuedReconstruction);
    QTimer::singleShot(1000, this, &MainWindow::maybeStartCtSmokeAutostart);
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
    for (auto it = motorAxes_.begin(); it != motorAxes_.end(); ++it)
    {
        const auto motorIt = config.motorAxes.constFind(it.key());
        if (motorIt == config.motorAxes.constEnd())
        {
            continue;
        }
        if (it->speedSpin)
        {
            it->speedSpin->setValue(motorIt.value().speed);
        }
        if (it->accelSpin)
        {
            it->accelSpin->setValue(motorIt.value().accel);
        }
    }
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
    cfg.motorAxes.clear();
    for (auto it = motorAxes_.cbegin(); it != motorAxes_.cend(); ++it)
    {
        MotorAxisConfig motorCfg;
        motorCfg.speed = it->speedSpin ? it->speedSpin->value() : 0.0;
        motorCfg.accel = it->accelSpin ? it->accelSpin->value() : 0.0;
        cfg.motorAxes.insert(it.key(), motorCfg);
    }
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
    recentLogLines_.append(text);
    while (recentLogLines_.size() > 200)
    {
        recentLogLines_.removeFirst();
    }

    QFile file(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("ct_workstation_run.log")));
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        QTextStream stream(&file);
        stream << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << " " << text << '\n';
    }
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

void MainWindow::browseCtSavePath()
{
    const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Select CT Save Path"), ctSavePathEdit_->text());
    if (!path.isEmpty())
    {
        ctSavePathEdit_->setText(path);
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
    connect(cameraWorker_, &PvcamCameraWorker::frameReady, this, [this](const CameraFrame& frame) {
        if (ctRunning_ && ctWaitingForFrame_)
        {
            enqueueCtFrame(frame);
            return;
        }
        updateCameraFrame(frame);
    });
    connect(cameraWorker_, &PvcamCameraWorker::logMessage, this, &MainWindow::appendLog);
    connect(cameraWorker_, &PvcamCameraWorker::captureFinished, this, [this](bool ok, const QString& message) {
        if (shuttingDown_) { return; }
        if (ctRunning_)
        {
            if (!ok)
            {
                finishCtAcquisition(false, QStringLiteral("Kinetix capture failed: %1").arg(message));
            }
            else
            {
                ctWaitingForCaptureFinish_ = false;
                QTimer::singleShot(0, this, &MainWindow::advanceCtAcquisition);
            }
            return;
        }
        const bool shouldRestartLive = cameraRestartLiveAfterStop_;
        cameraRestartLiveAfterStop_ = false;
        cameraLiveRunning_ = false;
        const bool completedSingleSnap = pendingSingleSnapIndex_ >= 0;
        const int completedSingleSnapIndex = pendingSingleSnapIndex_;
        pendingSingleSnapIndex_ = -1;
        cameraSnapButton_->setEnabled(true);
        cameraLiveButton_->setEnabled(true);
        cameraLiveButton_->setText(QStringLiteral("Open Live"));
        cameraApplyRoiButton_->setEnabled(true);
        cameraResetRoiButton_->setEnabled(true);
        if (ok && completedSingleSnap)
        {
            cameraSnapIndexSpin_->setValue(completedSingleSnapIndex);
        }
        appendLog(QStringLiteral("Kinetix capture %1: %2").arg(ok ? QStringLiteral("ok") : QStringLiteral("failed"), message));
        if (shouldRestartLive && ok)
        {
            toggleCameraLivePreview();
        }
    });
    cameraThread_.start();
    QMetaObject::invokeMethod(cameraWorker_, "warmOpen", Qt::QueuedConnection);

    ctWriter_ = new CtTiffWriter();
    ctWriter_->moveToThread(&ctWriterThread_);
    connect(ctWriter_, &CtTiffWriter::logMessage, this, &MainWindow::appendLog);
    connect(ctWriter_, &CtTiffWriter::queueChanged, this, [this](int queued, int saved) {
        if (ctQueueLabel_)
        {
            ctQueueLabel_->setText(QStringLiteral("Queue %1 / Saved %2").arg(queued).arg(saved));
        }
    });
    connect(ctWriter_, &CtTiffWriter::finished, this, [this](bool ok, const QString& message) {
        appendLog(message);
        if (ctRunning_)
        {
            finishCtAcquisition(ok, message);
        }
    });
    ctWriterThread_.start();
}

void MainWindow::snapCameraFrame()
{
    if (!cameraWorker_)
    {
        appendLog(QStringLiteral("Kinetix worker is unavailable."));
        return;
    }

    if (cameraLiveRunning_)
    {
        if (cameraSnapSaveInProgress_)
        {
            appendLog(QStringLiteral("Kinetix snap is still saving; wait for it to finish before taking another snap."));
            return;
        }
        if (lastCameraSaveFrame_.empty())
        {
            appendLog(QStringLiteral("No Kinetix live frame is available yet; wait for the first preview frame before Snap."));
            return;
        }
        const QString outputDir = cameraSaveDirEdit_->text();
        QDir().mkpath(outputDir);
        if (!QFileInfo(outputDir).isDir())
        {
            appendLog(QStringLiteral("Kinetix snap failed: output directory is unavailable."));
            return;
        }
        const int snapIndex = cameraSnapIndexSpin_->value();
        const QString framePath = cameraSnapPathForIndex(
            outputDir,
            cameraSnapPrefixEdit_->text(),
            cameraSnapPaddingSpin_->value(),
            snapIndex);
        const cv::Mat frameToSave = lastCameraSaveFrame_.clone();
        cameraSnapSaveInProgress_ = true;
        cameraSnapButton_->setEnabled(false);
        appendLog(QStringLiteral("Saving Kinetix live snap in background: %1.").arg(framePath));
        std::thread([this, framePath, frameToSave, snapIndex]() {
            const bool ok = cv::imwrite(framePath.toStdString(), frameToSave);
            QMetaObject::invokeMethod(this, [this, ok, framePath, snapIndex]() {
                if (shuttingDown_) { return; }
                cameraSnapSaveInProgress_ = false;
                cameraSnapButton_->setEnabled(true);
                if (ok)
                {
                    cameraSnapIndexSpin_->setValue(snapIndex);
                }
                appendLog(ok
                    ? QStringLiteral("Kinetix live snap saved: %1.").arg(framePath)
                    : QStringLiteral("Kinetix snap failed: could not save %1.").arg(framePath));
            }, Qt::QueuedConnection);
        }).detach();
        return;
    }

    lastRawCameraFrame_.release();
    lastCameraSaveFrame_.release();
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;
    cameraHistogramWidget_->resetWindow();
    cameraPreviewFitInitialized_ = false;
    cameraPreviewView_->setImage(QPixmap());
    cameraPreviewView_->resetTransform();
    cameraSnapButton_->setEnabled(false);
    cameraLiveButton_->setEnabled(false);
    cameraApplyRoiButton_->setEnabled(false);
    cameraResetRoiButton_->setEnabled(false);
    const QString outputDir = cameraSaveDirEdit_->text();
    QDir().mkpath(outputDir);
    if (!QFileInfo(outputDir).isDir())
    {
        appendLog(QStringLiteral("Kinetix snap failed: output directory is unavailable."));
        cameraSnapButton_->setEnabled(true);
        cameraLiveButton_->setEnabled(true);
        cameraApplyRoiButton_->setEnabled(true);
        cameraResetRoiButton_->setEnabled(true);
        return;
    }
    const int snapIndex = cameraSnapIndexSpin_->value();
    const QString framePath = cameraSnapPathForIndex(
        outputDir,
        cameraSnapPrefixEdit_->text(),
        cameraSnapPaddingSpin_->value(),
        snapIndex);
    pendingSingleSnapIndex_ = snapIndex;

    PvcamCaptureSettings settings;
    settings.frameCount = 1;
    settings.exposureValue = cameraExposureSpin_->value();
    settings.exposureUnit = cameraExposureUnitFromText(cameraExposureUnitCombo_->currentText());
    settings.roiX = cameraRoiXSpin_->value();
    settings.roiY = cameraRoiYSpin_->value();
    settings.roiWidth = cameraRoiWidthSpin_->value();
    settings.roiHeight = cameraRoiHeightSpin_->value();
    settings.queueCapacity = 1;
    settings.saveRaw = true;
    settings.outputDir = outputDir;
    settings.filePrefix = cameraSnapPrefixEdit_->text();
    settings.filePadding = cameraSnapPaddingSpin_->value();
    settings.singleFramePath = framePath;

    appendLog(QStringLiteral("Snapping one Kinetix frame to %1: ROI x=%2 y=%3 w=%4 h=%5.")
        .arg(framePath)
        .arg(settings.roiX)
        .arg(settings.roiY)
        .arg(settings.roiWidth)
        .arg(settings.roiHeight));
    QMetaObject::invokeMethod(
        cameraWorker_,
        "startCapture",
        Qt::QueuedConnection,
        Q_ARG(ct::PvcamCaptureSettings, settings));
}

void MainWindow::toggleCameraLivePreview()
{
    if (!cameraWorker_)
    {
        appendLog(QStringLiteral("Kinetix worker is unavailable."));
        return;
    }

    if (cameraLiveRunning_)
    {
        stopCameraLivePreview(false);
        return;
    }

    lastRawCameraFrame_.release();
    lastCameraSaveFrame_.release();
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;
    cameraHistogramWidget_->resetWindow();
    cameraPreviewFitInitialized_ = false;
    cameraPreviewView_->setImage(QPixmap());
    cameraPreviewView_->resetTransform();
    cameraSnapButton_->setEnabled(true);
    cameraLiveButton_->setEnabled(true);
    cameraLiveButton_->setText(QStringLiteral("Close Live"));
    cameraApplyRoiButton_->setEnabled(true);
    cameraResetRoiButton_->setEnabled(true);
    cameraLiveRunning_ = true;

    PvcamCaptureSettings settings;
    settings.frameCount = 1;
    settings.exposureValue = cameraExposureSpin_->value();
    settings.exposureUnit = cameraExposureUnitFromText(cameraExposureUnitCombo_->currentText());
    settings.roiX = cameraRoiXSpin_->value();
    settings.roiY = cameraRoiYSpin_->value();
    settings.roiWidth = cameraRoiWidthSpin_->value();
    settings.roiHeight = cameraRoiHeightSpin_->value();
    settings.queueCapacity = 1;
    settings.saveRaw = false;
    settings.previewOnly = true;
    settings.outputDir = cameraSaveDirEdit_->text();
    settings.previewIntervalMs = previewIntervalForRoi(settings.roiWidth, settings.roiHeight);

    appendLog(QStringLiteral("Opening Kinetix live preview: exposure=%1 %2, ROI x=%3 y=%4 w=%5 h=%6, UI interval=%7 ms.")
        .arg(settings.exposureValue)
        .arg(cameraExposureUnitCombo_->currentText())
        .arg(settings.roiX)
        .arg(settings.roiY)
        .arg(settings.roiWidth)
        .arg(settings.roiHeight)
        .arg(settings.previewIntervalMs));
    QMetaObject::invokeMethod(
        cameraWorker_,
        "startCapture",
        Qt::QueuedConnection,
        Q_ARG(ct::PvcamCaptureSettings, settings));
}

void MainWindow::stopCameraLivePreview(bool restartAfterStop)
{
    if (!cameraWorker_)
    {
        return;
    }
    cameraRestartLiveAfterStop_ = restartAfterStop;
    cameraLiveButton_->setEnabled(false);
    cameraSnapButton_->setEnabled(false);
    cameraApplyRoiButton_->setEnabled(false);
    cameraResetRoiButton_->setEnabled(false);
    QMetaObject::invokeMethod(cameraWorker_, "stopCapture", Qt::DirectConnection);
}

void MainWindow::applyCameraRoi()
{
    const int sensorWidth = 3200;
    const int sensorHeight = 3200;
    const int x = std::clamp(cameraRoiXSpin_->value(), 0, sensorWidth - 1);
    const int y = std::clamp(cameraRoiYSpin_->value(), 0, sensorHeight - 1);
    const int width = std::clamp(cameraRoiWidthSpin_->value(), 1, sensorWidth - x);
    const int height = std::clamp(cameraRoiHeightSpin_->value(), 1, sensorHeight - y);
    cameraRoiXSpin_->setValue(x);
    cameraRoiYSpin_->setValue(y);
    cameraRoiWidthSpin_->setValue(width);
    cameraRoiHeightSpin_->setValue(height);
    appendLog(QStringLiteral("Applied Kinetix ROI x=%1 y=%2 w=%3 h=%4.").arg(x).arg(y).arg(width).arg(height));
    if (cameraLiveRunning_)
    {
        stopCameraLivePreview(true);
    }
}

void MainWindow::resetCameraRoiFullFrame()
{
    cameraRoiXSpin_->setValue(0);
    cameraRoiYSpin_->setValue(0);
    cameraRoiWidthSpin_->setValue(3200);
    cameraRoiHeightSpin_->setValue(3200);
    applyCameraRoi();
}

void MainWindow::updateCameraFrame(const CameraFrame& frame)
{
    if (shuttingDown_ || frame.image.empty())
    {
        return;
    }

    ++cameraPreviewFrameCount_;
    lastCameraSaveFrame_ = frame.image.clone();
    const cv::Mat previewMat = makeCameraPreviewMat(frame.image);
    previewMat.convertTo(lastRawCameraFrame_, CV_32F);

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

void MainWindow::updateCameraFrameForCtPreview(const CameraFrame& frame)
{
    if (shuttingDown_ || frame.image.empty())
    {
        return;
    }

    ++cameraPreviewFrameCount_;
    const cv::Mat previewMat = makeCameraPreviewMat(frame.image);
    previewMat.convertTo(lastRawCameraFrame_, CV_32F);

    if (!currentCameraPreviewWindow_.isValid() || cameraPreviewFrameCount_ == 1)
    {
        currentCameraPreviewWindow_ = PreviewSliceRenderer::estimateRobustWindow(lastRawCameraFrame_);
    }

    if (cameraPreviewFrameCount_ == 1)
    {
        appendLog(QStringLiteral("CT preview first frame: %1x%2, bitDepth=%3, cvType=%4, stride=%5.")
            .arg(frame.width)
            .arg(frame.height)
            .arg(frame.bitDepth)
            .arg(frame.cvType)
            .arg(frame.strideBytes));
    }

    lastCameraPreviewRenderMs_ = frame.timestampMs > 0 ? frame.timestampMs : lastCameraPreviewRenderMs_ + 1;
    renderCameraPreviewWithCurrentWindow();
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

void MainWindow::startCtAcquisition()
{
    if (ctRunning_)
    {
        appendLog(QStringLiteral("CT acquisition is already running."));
        return;
    }
    if (!cameraWorker_ || !ctWriter_)
    {
        appendLog(QStringLiteral("CT acquisition failed: camera or writer is unavailable."));
        return;
    }
    if (!motorConnected_)
    {
        appendLog(QStringLiteral("CT acquisition failed: Kohzu is not connected."));
        return;
    }
    if (cameraLiveRunning_)
    {
        appendLog(QStringLiteral("Close Kinetix Live before starting CT acquisition."));
        return;
    }
    const QString savePath = ctSavePathEdit_->text().trimmed();
    if (savePath.isEmpty())
    {
        appendLog(QStringLiteral("CT acquisition failed: Save Path is empty."));
        return;
    }
    QDir().mkpath(savePath);
    QDir().mkpath(QDir(savePath).filePath(QStringLiteral("recon")));
    if (!QFileInfo(savePath).isDir())
    {
        appendLog(QStringLiteral("CT acquisition failed: Save Path is unavailable."));
        return;
    }
    if (!motorAxes_.contains(QStringLiteral("R")) || !motorAxes_.contains(QStringLiteral("Y")))
    {
        appendLog(QStringLiteral("CT acquisition failed: R/Y axes are unavailable."));
        return;
    }

    const MotorAxisUi& rAxis = motorAxes_[QStringLiteral("R")];
    const MotorAxisUi& yAxis = motorAxes_[QStringLiteral("Y")];
    ctSettings_ = {};
    ctSettings_.savePath = savePath;
    ctSettings_.beginningAngleDeg = ctBeginningAngleSpin_->value();
    ctSettings_.projectionCount = ctProjectionNumSpin_->value();
    ctSettings_.flatGap = ctFlatGapSpin_->value();
    ctSettings_.flatMoveDistanceUm = ctFlatMoveSpin_->value();
    ctSettings_.flatCount = ctFlatNumSpin_->value();
    ctSettings_.autoCor = ctAutoCorCheck_->isChecked();
    ctSettings_.padding = paddingSpin_->value();
    ctSettings_.tomoPrefix = tomoPrefixEdit_->text().isEmpty() ? QStringLiteral("Tomo_") : tomoPrefixEdit_->text();
    ctSettings_.flatPrefix = flatPrefixEdit_->text().isEmpty() ? QStringLiteral("Flat_") : flatPrefixEdit_->text();
    ctSettings_.extension = extCombo_->currentText().isEmpty() ? QStringLiteral("tiff") : extCombo_->currentText();
    ctSettings_.tiffThreads = ctTiffThreadsSpin_->value();
    ctSettings_.camera.frameCount = 1;
    ctSettings_.camera.exposureValue = cameraExposureSpin_->value();
    ctSettings_.camera.exposureUnit = cameraExposureUnitFromText(cameraExposureUnitCombo_->currentText());
    ctSettings_.camera.roiX = cameraRoiXSpin_->value();
    ctSettings_.camera.roiY = cameraRoiYSpin_->value();
    ctSettings_.camera.roiWidth = cameraRoiWidthSpin_->value();
    ctSettings_.camera.roiHeight = cameraRoiHeightSpin_->value();
    ctSettings_.camera.queueCapacity = 1;
    ctSettings_.camera.saveRaw = false;
    ctSettings_.camera.previewOnly = false;
    ctSettings_.camera.enqueueFrames = false;
    ctSettings_.camera.outputDir = savePath;
    ctSettings_.camera.previewIntervalMs = 0;
    ctSettings_.reconConfig = configFromUi();
    if (ctSettings_.reconConfig.io.reconMethod == ReconMethod::RealTimeFBP)
    {
        ctSettings_.reconConfig.io.reconMethod = ReconMethod::OfflineFBP;
    }
    ctSettings_.reconConfig.phase.enable = false;
    ctSettings_.reconConfig.geometry.corAuto = ctSettings_.autoCor;
    ctSettings_.reconConfig.io.watchDir = savePath;
    ctSettings_.reconConfig.io.saveDir = QDir(savePath).filePath(QStringLiteral("recon"));
    ctSettings_.reconConfig.io.tomoPrefix = ctSettings_.tomoPrefix;
    ctSettings_.reconConfig.io.flatPrefix = ctSettings_.flatPrefix;
    ctSettings_.reconConfig.io.inputPadding = ctSettings_.padding;
    ctSettings_.reconConfig.io.inputExt = ctSettings_.extension;
    ctSettings_.reconConfig.io.filePattern0 = numberedPath(QString(), ctSettings_.tomoPrefix,
        ctSettings_.padding, 0, ctSettings_.extension);
    ctSettings_.reconConfig.io.filePattern180 = QStringLiteral("alignment180.%1").arg(ctSettings_.extension);
    ctSettings_.reconConfig.geometry.totalAngles = ctSettings_.projectionCount;
    ctSettings_.reconConfig.geometry.volW = ctSettings_.camera.roiWidth;
    ctSettings_.reconConfig.geometry.volH = ctSettings_.camera.roiHeight;
    ctSettings_.reconConfig.geometry.angleOffsetDeg = static_cast<float>(ctSettings_.beginningAngleDeg);
    ctSettings_.rAxis = rAxis.axis;
    ctSettings_.yAxis = yAxis.axis;
    ctSettings_.rPulsePerDeg = rAxis.pulsePerUnit;
    ctSettings_.yPulsePerUm = yAxis.pulsePerUnit;

    const double endAngle = ctSettings_.beginningAngleDeg + 180.0;
    if (endAngle < rAxis.softMin || endAngle > rAxis.softMax)
    {
        appendLog(QStringLiteral("CT acquisition failed: final R angle %1 %2 exceeds soft limit [%3, %4].")
            .arg(endAngle, 0, 'f', 4)
            .arg(rAxis.unit)
            .arg(rAxis.softMin, 0, 'f', 4)
            .arg(rAxis.softMax, 0, 'f', 4));
        return;
    }
    const double yFlatTarget = yAxis.currentValue + ctSettings_.flatMoveDistanceUm;
    if (ctSettings_.flatCount > 0 && (yFlatTarget < yAxis.softMin || yFlatTarget > yAxis.softMax))
    {
        appendLog(QStringLiteral("CT acquisition failed: flat Y target %1 %2 exceeds soft limit [%3, %4].")
            .arg(yFlatTarget, 0, 'f', 4)
            .arg(yAxis.unit)
            .arg(yAxis.softMin, 0, 'f', 4)
            .arg(yAxis.softMax, 0, 'f', 4));
        return;
    }

    ctManifestPath_ = QDir(savePath).filePath(QStringLiteral("scan_manifest.json"));
    ctFlatGroups_.clear();
    ctProjectionIndex_ = 0;
    ctFlatIndex_ = 0;
    ctFlatFramesRemaining_ = 0;
    ctFlatAfterProjectionCount_ = 0;
    ctLastFlatAfterProjectionCount_ = -1;
    ctCurrentAngleDeg_ = ctSettings_.beginningAngleDeg;
    ctYIsOutForFlat_ = false;
    ctFinalFlatDone_ = false;
    ctStopRequested_ = false;
    ctWaitingForFrame_ = false;
    ctWaitingForCaptureFinish_ = false;
    ctWaitingForMotor_ = false;
    ctPendingFrameIsFlat_ = false;
    ctPendingFrameKind_ = CtFrameKind::Tomo;
    ctProjectionNeedsMove_ = false;
    ctCorReferencePending_ = ctSettings_.autoCor;
    ctCorReferenceCaptured_ = false;
    ctPendingMotorAction_.clear();
    ctRunning_ = true;

    ctStartButton_->setEnabled(false);
    ctStopButton_->setEnabled(true);
    ctStatusLabel_->setText(QStringLiteral("Running"));
    cameraSnapButton_->setEnabled(false);
    cameraLiveButton_->setEnabled(false);
    cameraApplyRoiButton_->setEnabled(false);
    cameraResetRoiButton_->setEnabled(false);
    cameraExposureSpin_->setEnabled(false);
    cameraExposureUnitCombo_->setEnabled(false);
    cameraRoiXSpin_->setEnabled(false);
    cameraRoiYSpin_->setEnabled(false);
    cameraRoiWidthSpin_->setEnabled(false);
    cameraRoiHeightSpin_->setEnabled(false);
    ctBeginningAngleSpin_->setEnabled(false);
    ctSavePathEdit_->setEnabled(false);
    ctProjectionNumSpin_->setEnabled(false);
    ctFlatGapSpin_->setEnabled(false);
    ctFlatMoveSpin_->setEnabled(false);
    ctFlatNumSpin_->setEnabled(false);
    ctAutoCorCheck_->setEnabled(false);
    ctTiffThreadsSpin_->setEnabled(false);
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;

    const int maxQueued = std::max(16, std::min(256, ctSettings_.tiffThreads * 4));
    QMetaObject::invokeMethod(ctWriter_, [writer = ctWriter_, threads = ctSettings_.tiffThreads, maxQueued]() {
        writer->configure(threads, maxQueued);
        writer->start();
    }, Qt::QueuedConnection);

    appendLog(QStringLiteral("CT acquisition started: %1 projections, begin=%2 deg, step=%3 deg, flats gap=%4 num=%5, auto COR=%6.")
        .arg(ctSettings_.projectionCount)
        .arg(ctSettings_.beginningAngleDeg, 0, 'f', 4)
        .arg(180.0 / std::max(1, ctSettings_.projectionCount), 0, 'f', 6)
        .arg(ctSettings_.flatGap)
        .arg(ctSettings_.flatCount)
        .arg(ctSettings_.autoCor ? QStringLiteral("on") : QStringLiteral("off")));
    ctWaitingForMotor_ = true;
    ctPendingMotorAction_ = ctSettings_.autoCor ? QStringLiteral("r_cor180") : QStringLiteral("r_start");
    const double firstTargetAngle = ctSettings_.autoCor
        ? ctSettings_.beginningAngleDeg + 180.0
        : ctSettings_.beginningAngleDeg;
    QMetaObject::invokeMethod(kohzuController_, "moveAbsolutePulse", Qt::QueuedConnection,
        Q_ARG(int, ctSettings_.rAxis),
        Q_ARG(qint64, static_cast<qint64>(std::llround(firstTargetAngle * ctSettings_.rPulsePerDeg))),
        Q_ARG(int, 0),
        Q_ARG(QString, ctSettings_.autoCor
            ? QStringLiteral("CT R COR 180 %1 deg").arg(firstTargetAngle, 0, 'f', 4)
            : QStringLiteral("CT R start %1 deg").arg(firstTargetAngle, 0, 'f', 4)));
}

void MainWindow::stopCtAcquisition()
{
    if (!ctRunning_)
    {
        return;
    }
    ctStopRequested_ = true;
    appendLog(QStringLiteral("Stopping CT acquisition."));
    if (cameraWorker_)
    {
        QMetaObject::invokeMethod(cameraWorker_, "stopCapture", Qt::QueuedConnection);
    }
    if (kohzuController_)
    {
        QMetaObject::invokeMethod(kohzuController_, "stopAllAxes", Qt::QueuedConnection);
    }
    if (ctWriter_)
    {
        QMetaObject::invokeMethod(ctWriter_, "stop", Qt::QueuedConnection);
    }
    finishCtAcquisition(false, QStringLiteral("CT acquisition stopped by user."));
}

void MainWindow::maybeStartCtSmokeAutostart()
{
    if (qEnvironmentVariableIntValue("CT_SMOKE_AUTOSTART") != 1)
    {
        return;
    }
    if (ctRunning_ || controller_.isBusy())
    {
        return;
    }

    const bool cameraReady = cameraStateLabel_ && cameraStateLabel_->text().contains(QStringLiteral("Ready"), Qt::CaseInsensitive);
    const bool axesReady = motorAxes_.contains(QStringLiteral("R"))
        && motorAxes_.contains(QStringLiteral("Y"))
        && motorAxes_[QStringLiteral("R")].hasPosition
        && motorAxes_[QStringLiteral("Y")].hasPosition;
    if ((!motorConnected_ || !cameraReady || !axesReady) && ctSmokeAutostartAttempts_ < 60)
    {
        ++ctSmokeAutostartAttempts_;
        if ((ctSmokeAutostartAttempts_ % 5) == 1)
        {
            appendLog(QStringLiteral("CT smoke autostart waiting: motorConnected=%1, cameraReady=%2, axesReady=%3.")
                .arg(motorConnected_ ? QStringLiteral("true") : QStringLiteral("false"))
                .arg(cameraReady ? QStringLiteral("true") : QStringLiteral("false"))
                .arg(axesReady ? QStringLiteral("true") : QStringLiteral("false")));
        }
        QTimer::singleShot(1000, this, &MainWindow::maybeStartCtSmokeAutostart);
        return;
    }
    if (!motorConnected_ || !cameraReady || !axesReady)
    {
        appendLog(QStringLiteral("CT smoke autostart failed: hardware did not become ready."));
        return;
    }

    const QString savePath = qEnvironmentVariable("CT_SMOKE_SAVE_PATH",
        QDir(QStringLiteral("E:/2026_Chun/2026_06_09")).filePath(
            QStringLiteral("ct_acq_smoke_%1").arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss")))));

    cameraRoiXSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_ROI_X"));
    cameraRoiYSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_ROI_Y"));
    cameraRoiWidthSpin_->setValue(std::max(1, qEnvironmentVariableIntValue("CT_SMOKE_ROI_W") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_ROI_W")
            : 512));
    cameraRoiHeightSpin_->setValue(std::max(1, qEnvironmentVariableIntValue("CT_SMOKE_ROI_H") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_ROI_H")
            : 512));
    cameraExposureSpin_->setValue(qEnvironmentVariable("CT_SMOKE_EXPOSURE_MS").isEmpty()
            ? 10.0
            : qEnvironmentVariable("CT_SMOKE_EXPOSURE_MS").toDouble());
    cameraExposureUnitCombo_->setCurrentText(QStringLiteral("ms"));

    ctSavePathEdit_->setText(savePath);
    ctBeginningAngleSpin_->setValue(qEnvironmentVariable("CT_SMOKE_BEGIN_ANGLE").isEmpty()
            ? -180.0
            : qEnvironmentVariable("CT_SMOKE_BEGIN_ANGLE").toDouble());
    ctProjectionNumSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_PROJECTIONS") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_PROJECTIONS")
            : 5);
    ctFlatGapSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_FLAT_GAP") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_FLAT_GAP")
            : 2);
    ctFlatMoveSpin_->setValue(qEnvironmentVariable("CT_SMOKE_FLAT_MOVE_UM").isEmpty()
            ? 10.0
            : qEnvironmentVariable("CT_SMOKE_FLAT_MOVE_UM").toDouble());
    ctFlatNumSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_FLAT_NUM") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_FLAT_NUM")
            : 1);
    ctAutoCorCheck_->setChecked(true);
    ctTiffThreadsSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_TIFF_THREADS") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_TIFF_THREADS")
            : 4);

    correctionModeCombo_->setCurrentText(toString(CorrectionMode::Flat));
    reconMethodCombo_->setCurrentText(toString(ReconMethod::OfflineFBP));
    batchSpin_->setValue(4);
    zChunkSpin_->setValue(16);
    tiffThreadsSpin_->setValue(4);

    appendLog(QStringLiteral("CT smoke autostart: path=%1, ROI=%2x%3, projections=%4, flatGap=%5, flatMove=%6 um.")
        .arg(savePath)
        .arg(cameraRoiWidthSpin_->value())
        .arg(cameraRoiHeightSpin_->value())
        .arg(ctProjectionNumSpin_->value())
        .arg(ctFlatGapSpin_->value())
        .arg(ctFlatMoveSpin_->value(), 0, 'f', 4));
    startCtAcquisition();
}

void MainWindow::advanceCtAcquisition()
{
    if (!ctRunning_ || ctWaitingForFrame_ || ctWaitingForCaptureFinish_ || ctWaitingForMotor_)
    {
        return;
    }
    if (ctCorReferencePending_)
    {
        requestCtFrame(CtFrameKind::Cor180);
        return;
    }
    if (ctProjectionNeedsMove_)
    {
        ctProjectionNeedsMove_ = false;
        const double nextAngle = ctSettings_.beginningAngleDeg
            + (static_cast<double>(ctProjectionIndex_) * 180.0 / static_cast<double>(std::max(1, ctSettings_.projectionCount)));
        const qint64 pulse = static_cast<qint64>(std::llround(nextAngle * ctSettings_.rPulsePerDeg));
        ctCurrentAngleDeg_ = nextAngle;
        ctWaitingForMotor_ = true;
        ctPendingMotorAction_ = QStringLiteral("r_move");
        QMetaObject::invokeMethod(kohzuController_, "moveAbsolutePulse", Qt::QueuedConnection,
            Q_ARG(int, ctSettings_.rAxis), Q_ARG(qint64, pulse), Q_ARG(int, 0),
            Q_ARG(QString, QStringLiteral("CT R to %1 deg").arg(nextAngle, 0, 'f', 4)));
        return;
    }
    if (ctStopRequested_)
    {
        finishCtAcquisition(false, QStringLiteral("CT acquisition stopped."));
        return;
    }
    if (ctFlatFramesRemaining_ > 0)
    {
        requestCtFrame(CtFrameKind::Flat);
        return;
    }
    if (ctYIsOutForFlat_)
    {
        const qint64 pulse = static_cast<qint64>(std::llround(-ctSettings_.flatMoveDistanceUm * ctSettings_.yPulsePerUm));
        if (pulse != 0)
        {
            ctWaitingForMotor_ = true;
            ctPendingMotorAction_ = QStringLiteral("flat_return");
            QMetaObject::invokeMethod(kohzuController_, "jogRelativePulse", Qt::QueuedConnection,
                Q_ARG(int, ctSettings_.yAxis), Q_ARG(qint64, pulse), Q_ARG(int, 0), Q_ARG(QString, QStringLiteral("CT Y return")));
            return;
        }
        ctYIsOutForFlat_ = false;
    }
    const bool needStartFlat = ctSettings_.flatCount > 0 && ctProjectionIndex_ == 0 && ctFlatIndex_ == 0;
    const bool needIntervalFlat = ctSettings_.flatCount > 0
        && ctSettings_.flatGap > 0
        && ctProjectionIndex_ > 0
        && ctProjectionIndex_ < ctSettings_.projectionCount
        && ctLastFlatAfterProjectionCount_ != ctProjectionIndex_
        && (ctProjectionIndex_ % ctSettings_.flatGap) == 0;
    const bool needFinalFlat = ctSettings_.flatCount > 0 && ctProjectionIndex_ >= ctSettings_.projectionCount && !ctFinalFlatDone_;
    if (needStartFlat || needIntervalFlat || needFinalFlat)
    {
        ctFlatFramesRemaining_ = ctSettings_.flatCount;
        ctFlatAfterProjectionCount_ = ctProjectionIndex_;
        ctLastFlatAfterProjectionCount_ = ctProjectionIndex_;
        if (needFinalFlat)
        {
            ctFinalFlatDone_ = true;
        }
        CtFlatGroup group;
        group.label = needStartFlat ? QStringLiteral("start") : (needFinalFlat ? QStringLiteral("end") : QStringLiteral("interval"));
        group.afterProjectionCount = ctFlatAfterProjectionCount_;
        group.firstFlatIndex = ctFlatIndex_;
        group.flatCount = ctSettings_.flatCount;
        group.yMoveUm = ctSettings_.flatMoveDistanceUm;
        ctFlatGroups_.append(group);
        const qint64 pulse = static_cast<qint64>(std::llround(ctSettings_.flatMoveDistanceUm * ctSettings_.yPulsePerUm));
        if (pulse != 0)
        {
            ctWaitingForMotor_ = true;
            ctPendingMotorAction_ = QStringLiteral("flat_out");
            QMetaObject::invokeMethod(kohzuController_, "jogRelativePulse", Qt::QueuedConnection,
                Q_ARG(int, ctSettings_.yAxis), Q_ARG(qint64, pulse), Q_ARG(int, 0), Q_ARG(QString, QStringLiteral("CT Y flat out")));
            return;
        }
        ctYIsOutForFlat_ = true;
        QTimer::singleShot(0, this, &MainWindow::advanceCtAcquisition);
        return;
    }
    if (ctProjectionIndex_ < ctSettings_.projectionCount)
    {
        requestCtFrame(CtFrameKind::Tomo);
        return;
    }
    QMetaObject::invokeMethod(ctWriter_, "finish", Qt::QueuedConnection);
}

void MainWindow::requestCtFrame(CtFrameKind kind)
{
    ctWaitingForFrame_ = true;
    ctWaitingForCaptureFinish_ = true;
    ctPendingFrameKind_ = kind;
    ctPendingFrameIsFlat_ = kind == CtFrameKind::Flat;
    if (kind == CtFrameKind::Cor180)
    {
        ctStatusLabel_->setText(QStringLiteral("COR 180"));
    }
    else
    {
        ctStatusLabel_->setText(kind == CtFrameKind::Flat
            ? QStringLiteral("Flat %1").arg(ctFlatIndex_)
            : QStringLiteral("Tomo %1/%2").arg(ctProjectionIndex_ + 1).arg(ctSettings_.projectionCount));
    }
    PvcamCaptureSettings settings = ctSettings_.camera;
    QMetaObject::invokeMethod(cameraWorker_, "startCapture", Qt::QueuedConnection, Q_ARG(ct::PvcamCaptureSettings, settings));
}

void MainWindow::enqueueCtFrame(const CameraFrame& frame)
{
    if (!ctRunning_)
    {
        return;
    }
    ctWaitingForFrame_ = false;
    updateCameraFrameForCtPreview(frame);
    const CtFrameKind kind = ctPendingFrameKind_;
    const bool flat = kind == CtFrameKind::Flat;
    const int fileIndex = flat ? ctFlatIndex_ : ctProjectionIndex_;
    CtFrameSaveItem item;
    item.filePath = kind == CtFrameKind::Cor180
        ? QDir(ctSettings_.savePath).filePath(ctSettings_.reconConfig.io.filePattern180)
        : numberedPath(ctSettings_.savePath, flat ? ctSettings_.flatPrefix : ctSettings_.tomoPrefix,
            ctSettings_.padding, fileIndex, ctSettings_.extension);
    item.image = frame.image;
    QMetaObject::invokeMethod(ctWriter_, "enqueue", Qt::BlockingQueuedConnection, Q_ARG(ct::CtFrameSaveItem, item));

    if (kind == CtFrameKind::Cor180)
    {
        ctCorReferencePending_ = false;
        ctCorReferenceCaptured_ = true;
        ctWaitingForMotor_ = true;
        ctPendingMotorAction_ = QStringLiteral("r_start");
        QMetaObject::invokeMethod(kohzuController_, "moveAbsolutePulse", Qt::QueuedConnection,
            Q_ARG(int, ctSettings_.rAxis),
            Q_ARG(qint64, static_cast<qint64>(std::llround(ctSettings_.beginningAngleDeg * ctSettings_.rPulsePerDeg))),
            Q_ARG(int, 0),
            Q_ARG(QString, QStringLiteral("CT R start %1 deg").arg(ctSettings_.beginningAngleDeg, 0, 'f', 4)));
        return;
    }

    if (flat)
    {
        ++ctFlatIndex_;
        --ctFlatFramesRemaining_;
        return;
    }
    ++ctProjectionIndex_;
    ctProjectionNeedsMove_ = true;
}

void MainWindow::finishCtAcquisition(bool ok, const QString& message)
{
    if (!ctRunning_)
    {
        return;
    }
    ctRunning_ = false;
    ctWaitingForFrame_ = false;
    ctWaitingForMotor_ = false;
    ctStatusLabel_->setText(ok ? QStringLiteral("Completed") : QStringLiteral("Failed"));
    ctStartButton_->setEnabled(true);
    ctStopButton_->setEnabled(false);
    cameraSnapButton_->setEnabled(true);
    cameraLiveButton_->setEnabled(true);
    cameraApplyRoiButton_->setEnabled(true);
    cameraResetRoiButton_->setEnabled(true);
    cameraExposureSpin_->setEnabled(true);
    cameraExposureUnitCombo_->setEnabled(true);
    cameraRoiXSpin_->setEnabled(true);
    cameraRoiYSpin_->setEnabled(true);
    cameraRoiWidthSpin_->setEnabled(true);
    cameraRoiHeightSpin_->setEnabled(true);
    ctBeginningAngleSpin_->setEnabled(true);
    ctSavePathEdit_->setEnabled(true);
    ctProjectionNumSpin_->setEnabled(true);
    ctFlatGapSpin_->setEnabled(true);
    ctFlatMoveSpin_->setEnabled(true);
    ctFlatNumSpin_->setEnabled(true);
    ctAutoCorCheck_->setEnabled(true);
    ctTiffThreadsSpin_->setEnabled(true);
    writeCtManifest(ok, ok ? QString() : message);
    appendLog(ok ? QStringLiteral("CT acquisition completed.") : QStringLiteral("CT acquisition failed/stopped: %1").arg(message));
    if (ok)
    {
        enqueueCtReconJob();
        refreshReconQueue();
        startNextQueuedReconstruction();
    }
}

void MainWindow::writeCtManifest(bool completed, const QString& errorMessage)
{
    if (ctManifestPath_.isEmpty())
    {
        return;
    }
    QJsonObject root;
    root["TYPE"] = "CT_STREAM_SCAN_MANIFEST";
    root["VERSION"] = 2;
    root["COMPLETED"] = completed;
    root["ERROR"] = errorMessage;
    root["CREATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);

    QJsonObject acq;
    acq["BEGINNING_ANGLE_DEG"] = ctSettings_.beginningAngleDeg;
    acq["END_ANGLE_DEG"] = ctSettings_.beginningAngleDeg + 180.0;
    acq["ANGLE_STEP_DEG"] = 180.0 / static_cast<double>(std::max(1, ctSettings_.projectionCount));
    acq["PROJECTION_COUNT"] = ctSettings_.projectionCount;
    acq["FLAT_GAP"] = ctSettings_.flatGap;
    acq["FLAT_MOVE_DISTANCE_UM"] = ctSettings_.flatMoveDistanceUm;
    acq["FLAT_NUM"] = ctSettings_.flatCount;
    acq["IMAGE_WIDTH"] = ctSettings_.camera.roiWidth;
    acq["IMAGE_HEIGHT"] = ctSettings_.camera.roiHeight;
    acq["AUTO_COR"] = ctSettings_.autoCor;
    acq["COR_REFERENCE_CAPTURED"] = ctCorReferenceCaptured_;
    acq["COR_SHIFT"] = ctSettings_.reconConfig.geometry.corShift;
    root["ACQUISITION"] = acq;
    root["CAMERA"] = pvcamSettingsToJson(ctSettings_.camera);

    QJsonObject io;
    io["TOMO_PREFIX"] = ctSettings_.tomoPrefix;
    io["FLAT_PREFIX"] = ctSettings_.flatPrefix;
    io["DARK_PREFIX"] = ctSettings_.reconConfig.io.darkPrefix;
    io["INPUT_PADDING"] = ctSettings_.padding;
    io["INPUT_EXT"] = ctSettings_.extension;
    io["FILE_PATTERN_0"] = ctSettings_.reconConfig.io.filePattern0;
    io["FILE_PATTERN_180"] = ctSettings_.reconConfig.io.filePattern180;
    root["IO"] = io;
    root["FLAT_GROUPS"] = flatGroupsToJson(ctFlatGroups_);

    QJsonObject motor;
    motor["R_AXIS"] = ctSettings_.rAxis;
    motor["R_PULSE_PER_DEG"] = ctSettings_.rPulsePerDeg;
    motor["Y_AXIS"] = ctSettings_.yAxis;
    motor["Y_PULSE_PER_UM"] = ctSettings_.yPulsePerUm;
    root["MOTOR"] = motor;
    if (writeJsonFileAtomic(ctManifestPath_, root))
    {
        appendLog(QStringLiteral("CT manifest written: %1").arg(ctManifestPath_));
    }
    else
    {
        appendLog(QStringLiteral("Failed to write CT manifest: %1").arg(ctManifestPath_));
    }
}

void MainWindow::enqueueCtReconJob()
{
    const QString pendingDir = QDir(reconQueueRoot()).filePath(QStringLiteral("pending"));
    QDir().mkpath(pendingDir);
    const QString base = sanitizeQueueName(QFileInfo(ctSettings_.savePath).fileName());
    int maxIndex = 0;
    const QStringList existing = QDir(pendingDir).entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
    for (const QString& fileName : existing)
    {
        maxIndex = std::max(maxIndex, queueIndexFromFileName(fileName));
    }
    if (maxIndex == std::numeric_limits<int>::max())
    {
        maxIndex = existing.size() * 10;
    }
    const int queueIndex = maxIndex + 10;
    const QString jobPath = QDir(pendingDir).filePath(queueFileName(queueIndex, base));
    QJsonObject job = appConfigToJobObject(ctSettings_.reconConfig);
    job["TYPE"] = "CT_STREAM_RECON_JOB";
    job["VERSION"] = 2;
    job["STATUS"] = "pending";
    job["QUEUE_INDEX"] = queueIndex;
    job["CREATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    job["PROJECTION_DIR"] = ctSettings_.savePath;
    job["OUTPUT_DIR"] = ctSettings_.reconConfig.io.saveDir;
    const QJsonObject jobParams = appConfigToJobObject(ctSettings_.reconConfig);
    for (auto it = jobParams.begin(); it != jobParams.end(); ++it)
    {
        job[it.key()] = it.value();
    }
    if (writeJsonFileAtomic(jobPath, job))
    {
        appendLog(QStringLiteral("Recon job queued: %1").arg(jobPath));
    }
    else
    {
        appendLog(QStringLiteral("Failed to enqueue recon job: %1").arg(jobPath));
    }
}

QString MainWindow::reconQueueRoot() const
{
    return QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("recon_queue"));
}

void MainWindow::ensureReconQueueDirs()
{
    QDir().mkpath(QDir(reconQueueRoot()).filePath(QStringLiteral("pending")));
    QDir().mkpath(QDir(reconQueueRoot()).filePath(QStringLiteral("running")));
    QDir().mkpath(QDir(reconQueueRoot()).filePath(QStringLiteral("done")));
    QDir().mkpath(QDir(reconQueueRoot()).filePath(QStringLiteral("failed")));
}

void MainWindow::recoverStaleRunningReconJobs()
{
    const QString runningDirPath = QDir(reconQueueRoot()).filePath(QStringLiteral("running"));
    const QString failedDirPath = QDir(reconQueueRoot()).filePath(QStringLiteral("failed"));
    QDir runningDir(runningDirPath);
    QDir().mkpath(failedDirPath);
    const QStringList files = runningDir.entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
    for (const QString& fileName : files)
    {
        const QString path = runningDir.filePath(fileName);
        QJsonObject job;
        readJsonFile(path, &job);
        job["STATUS"] = "failed";
        job["FAILED_STAGE"] = "JobParse";
        job["ERROR_CODE"] = "StaleRunningJob";
        job["ERROR_MESSAGE"] = "Job was left in running/ from a previous application exit.";
        job["FINISHED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        writeJsonFileAtomic(path, job);
        const QString target = QDir(failedDirPath).filePath(fileName);
        QFile::remove(target);
        QFile::rename(path, target);
        appendLog(QStringLiteral("Recovered stale running recon job to failed: %1").arg(target));
    }
}

void MainWindow::scheduleReconQueueRefresh()
{
    if (reconQueueRefreshTimer_)
    {
        reconQueueRefreshTimer_->start();
    }
}

void MainWindow::refreshReconQueue()
{
    if (!reconQueueList_)
    {
        return;
    }
    const QString selectedPath = reconQueueList_->currentItem()
        ? reconQueueList_->currentItem()->data(Qt::UserRole).toString()
        : QString();
    reconQueueList_->clear();
    const QString root = reconQueueRoot();
    const QStringList buckets = { QStringLiteral("running"), QStringLiteral("pending"), QStringLiteral("failed"), QStringLiteral("done") };
    for (const QString& bucket : buckets)
    {
        const QDir dir(QDir(root).filePath(bucket));
        const QStringList files = dir.entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
        for (const QString& file : files)
        {
            const QString path = dir.filePath(file);
            QString status = bucket;
            QJsonObject job;
            if (readJsonFile(path, &job))
            {
                status = job.value(QStringLiteral("STATUS")).toString(bucket);
            }
            auto* item = new QListWidgetItem(QStringLiteral("%1/%2").arg(bucket, file), reconQueueList_);
            item->setData(Qt::UserRole, path);
            item->setData(Qt::UserRole + 1, bucket);
            item->setToolTip(QStringLiteral("%1\nSTATUS=%2").arg(path, status));
            if (path == selectedPath)
            {
                item->setSelected(true);
                reconQueueList_->setCurrentItem(item);
            }
        }
    }
}

void MainWindow::normalizePendingReconQueue(bool scheduleRefresh)
{
    const QString pendingDirPath = QDir(reconQueueRoot()).filePath(QStringLiteral("pending"));
    QDir pendingDir(pendingDirPath);
    QStringList files = pendingDir.entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
    std::sort(files.begin(), files.end(), [](const QString& a, const QString& b) {
        const int ia = queueIndexFromFileName(a);
        const int ib = queueIndexFromFileName(b);
        if (ia != ib)
        {
            return ia < ib;
        }
        return a < b;
    });

    QVector<QPair<QString, QJsonObject>> jobs;
    jobs.reserve(files.size());
    int nextIndex = 10;
    for (const QString& file : files)
    {
        const QString oldPath = pendingDir.filePath(file);
        QJsonObject job;
        QString error;
        if (!readJsonFile(oldPath, &job, &error))
        {
            appendLog(QStringLiteral("Pending queue normalize skipped invalid job: %1").arg(error));
            continue;
        }
        job["STATUS"] = "pending";
        job["QUEUE_INDEX"] = nextIndex;
        jobs.append({ oldPath, job });
        nextIndex += 10;
    }
    QVector<QPair<QString, QJsonObject>> stagedJobs;
    stagedJobs.reserve(jobs.size());
    for (const auto& pair : jobs)
    {
        const QString stagedPath = pair.first + QStringLiteral(".renaming");
        QFile::remove(stagedPath);
        if (!QFile::rename(pair.first, stagedPath))
        {
            appendLog(QStringLiteral("Failed to stage pending job rename: %1").arg(pair.first));
            continue;
        }
        stagedJobs.append({ stagedPath, pair.second });
    }
    for (const auto& pair : stagedJobs)
    {
        const QJsonObject job = pair.second;
        const int queueIndex = job.value(QStringLiteral("QUEUE_INDEX")).toInt(10);
        const QString newPath = pendingDir.filePath(queueFileName(queueIndex, queueBaseNameFromPath(pair.first)));
        writeJsonFileAtomic(newPath, job);
        QFile::remove(pair.first);
    }
    if (scheduleRefresh)
    {
        scheduleReconQueueRefresh();
    }
}

void MainWindow::moveSelectedReconQueueJob(int direction)
{
    if (!reconQueueList_ || direction == 0)
    {
        return;
    }
    QListWidgetItem* selected = reconQueueList_->currentItem();
    if (!selected || selected->data(Qt::UserRole + 1).toString() != QStringLiteral("pending"))
    {
        appendLog(QStringLiteral("Only pending recon jobs can be reordered."));
        return;
    }
    const QString selectedPath = selected->data(Qt::UserRole).toString();
    const QString selectedBase = queueBaseNameFromPath(selectedPath);
    const QString pendingDirPath = QDir(reconQueueRoot()).filePath(QStringLiteral("pending"));
    QDir pendingDir(pendingDirPath);
    QStringList files = pendingDir.entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
    std::sort(files.begin(), files.end(), [](const QString& a, const QString& b) {
        const int ia = queueIndexFromFileName(a);
        const int ib = queueIndexFromFileName(b);
        return ia == ib ? a < b : ia < ib;
    });
    const QString selectedFile = QFileInfo(selectedPath).fileName();
    const int index = files.indexOf(selectedFile);
    const int target = index + direction;
    if (index < 0 || target < 0 || target >= files.size())
    {
        return;
    }
    files.swapItemsAt(index, target);

    QVector<QPair<QString, QJsonObject>> jobs;
    jobs.reserve(files.size());
    for (const QString& file : files)
    {
        QJsonObject job;
        if (readJsonFile(pendingDir.filePath(file), &job))
        {
            jobs.append({ pendingDir.filePath(file), job });
        }
    }
    QVector<QPair<QString, QJsonObject>> stagedJobs;
    stagedJobs.reserve(jobs.size());
    for (const auto& pair : jobs)
    {
        const QString stagedPath = pair.first + QStringLiteral(".reorder");
        QFile::remove(stagedPath);
        if (QFile::rename(pair.first, stagedPath))
        {
            stagedJobs.append({ stagedPath, pair.second });
        }
        else
        {
            appendLog(QStringLiteral("Failed to stage pending reorder: %1").arg(pair.first));
        }
    }
    int nextIndex = 10;
    QString restoredSelection;
    for (const auto& pair : stagedJobs)
    {
        QJsonObject job = pair.second;
        job["STATUS"] = "pending";
        job["QUEUE_INDEX"] = nextIndex;
        const QString oldBase = queueBaseNameFromPath(pair.first);
        const QString newPath = pendingDir.filePath(queueFileName(nextIndex, oldBase));
        writeJsonFileAtomic(newPath, job);
        QFile::remove(pair.first);
        if (queueBaseNameFromPath(pair.first) == selectedBase)
        {
            restoredSelection = newPath;
        }
        nextIndex += 10;
    }
    refreshReconQueue();
    if (!restoredSelection.isEmpty())
    {
        for (int i = 0; i < reconQueueList_->count(); ++i)
        {
            QListWidgetItem* item = reconQueueList_->item(i);
            if (item && item->data(Qt::UserRole).toString() == restoredSelection)
            {
                reconQueueList_->setCurrentItem(item);
                break;
            }
        }
    }
}

void MainWindow::removeSelectedReconQueueJob()
{
    if (!reconQueueList_)
    {
        return;
    }
    QListWidgetItem* selected = reconQueueList_->currentItem();
    if (!selected || selected->data(Qt::UserRole + 1).toString() != QStringLiteral("pending"))
    {
        appendLog(QStringLiteral("Only pending recon jobs can be removed."));
        return;
    }
    const QString sourcePath = selected->data(Qt::UserRole).toString();
    QJsonObject job;
    readJsonFile(sourcePath, &job);
    job["STATUS"] = "cancelled";
    job["FAILED_STAGE"] = "UserStopped";
    job["ERROR_CODE"] = "UserCancelled";
    job["ERROR_MESSAGE"] = "Pending reconstruction job removed by user.";
    job["FINISHED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    writeJsonFileAtomic(sourcePath, job);

    const QString failedDir = QDir(reconQueueRoot()).filePath(QStringLiteral("failed"));
    QDir().mkpath(failedDir);
    const QString targetPath = QDir(failedDir).filePath(QFileInfo(sourcePath).fileName());
    QFile::remove(targetPath);
    if (!QFile::rename(sourcePath, targetPath))
    {
        appendLog(QStringLiteral("Failed to remove pending recon job: %1").arg(sourcePath));
        return;
    }
    appendLog(QStringLiteral("Pending recon job cancelled: %1").arg(targetPath));
    normalizePendingReconQueue(true);
}

void MainWindow::startNextQueuedReconstruction()
{
    if (controller_.isBusy() || !activeQueueJobPath_.isEmpty())
    {
        return;
    }
    normalizePendingReconQueue(false);
    const QDir pendingDir(QDir(reconQueueRoot()).filePath(QStringLiteral("pending")));
    const QStringList files = pendingDir.entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
    if (files.isEmpty())
    {
        refreshReconQueue();
        return;
    }
    startQueuedReconstruction(pendingDir.filePath(files.first()));
}

bool MainWindow::startQueuedReconstruction(const QString& jobPath)
{
    QJsonObject job;
    QString error;
    if (!readJsonFile(jobPath, &job, &error))
    {
        appendLog(QStringLiteral("Failed to open recon job: %1").arg(error));
        const QJsonObject emptyJob;
        const QString resultPath = writeReconResultJson(jobPath, emptyJob, QStringLiteral("failed"), QStringLiteral("JobParse"),
            QStringLiteral("InvalidJobJson"), error, {}, {}, {}, QDateTime::currentDateTime().toString(Qt::ISODate),
            0.0, 0.0, false, 0.0);
        moveJobToFailed(jobPath, emptyJob, resultPath, QStringLiteral("JobParse"), QStringLiteral("InvalidJobJson"),
            error, reconQueueRoot());
        return false;
    }
    QString projectionDir = projectionDirFromJob(job);
    QString outputDir = job.value(QStringLiteral("OUTPUT_DIR")).toString();
    if (projectionDir.isEmpty())
    {
        const QString resultPath = writeReconResultJson(jobPath, job, QStringLiteral("failed"), QStringLiteral("JobParse"),
            QStringLiteral("MissingProjectionDir"), QStringLiteral("Recon job has no PROJECTION_DIR."),
            projectionDir, outputDir, {}, QDateTime::currentDateTime().toString(Qt::ISODate),
            0.0, 0.0, false, 0.0);
        moveJobToFailed(jobPath, job, resultPath, QStringLiteral("JobParse"), QStringLiteral("MissingProjectionDir"),
            QStringLiteral("Recon job has no PROJECTION_DIR."), reconQueueRoot());
        return false;
    }
    const QString manifestPath = manifestPathForProjectionDir(projectionDir);
    QJsonObject manifest;
    if (!readJsonFile(manifestPath, &manifest, &error))
    {
        const QString resultPath = writeReconResultJson(jobPath, job, QStringLiteral("failed"), QStringLiteral("ManifestLoad"),
            QStringLiteral("ManifestMissing"), error, projectionDir, outputDir, {},
            QDateTime::currentDateTime().toString(Qt::ISODate), 0.0, 0.0, false, 0.0);
        moveJobToFailed(jobPath, job, resultPath, QStringLiteral("ManifestLoad"), QStringLiteral("ManifestMissing"),
            error, reconQueueRoot());
        return false;
    }
    if (!manifest.value(QStringLiteral("COMPLETED")).toBool(false))
    {
        const QString message = QStringLiteral("scan_manifest.json COMPLETED=false.");
        const QString resultPath = writeReconResultJson(jobPath, job, QStringLiteral("failed"), QStringLiteral("ManifestValidate"),
            QStringLiteral("ManifestIncomplete"), QStringLiteral("scan_manifest.json COMPLETED=false."),
            projectionDir, outputDir, {}, QDateTime::currentDateTime().toString(Qt::ISODate),
            0.0, 0.0, false, 0.0);
        moveJobToFailed(jobPath, job, resultPath, QStringLiteral("ManifestValidate"),
            QStringLiteral("ManifestIncomplete"), message, reconQueueRoot());
        return false;
    }

    AppConfig cfg = AppConfig::fromJsonObject(manifest.value(QStringLiteral("RECON_CONFIG")).toObject());
    if (manifest.value(QStringLiteral("RECON_CONFIG")).toObject().isEmpty())
    {
        cfg = configFromUi();
    }
    applyJobObjectToConfig(job, cfg);
    const QJsonObject io = manifest.value(QStringLiteral("IO")).toObject();
    const QJsonObject acq = manifest.value(QStringLiteral("ACQUISITION")).toObject();
    cfg.io.watchDir = projectionDir;
    cfg.io.saveDir = outputDir.isEmpty() ? QDir(projectionDir).filePath(QStringLiteral("recon")) : outputDir;
    cfg.io.tomoPrefix = io.value(QStringLiteral("TOMO_PREFIX")).toString(cfg.io.tomoPrefix);
    cfg.io.flatPrefix = io.value(QStringLiteral("FLAT_PREFIX")).toString(cfg.io.flatPrefix);
    cfg.io.darkPrefix = io.value(QStringLiteral("DARK_PREFIX")).toString(cfg.io.darkPrefix);
    cfg.io.inputPadding = io.value(QStringLiteral("INPUT_PADDING")).toInt(cfg.io.inputPadding);
    cfg.io.inputExt = io.value(QStringLiteral("INPUT_EXT")).toString(cfg.io.inputExt);
    cfg.io.filePattern0 = io.value(QStringLiteral("FILE_PATTERN_0")).toString(cfg.io.filePattern0);
    cfg.io.filePattern180 = io.value(QStringLiteral("FILE_PATTERN_180")).toString(cfg.io.filePattern180);
    cfg.geometry.totalAngles = acq.value(QStringLiteral("PROJECTION_COUNT")).toInt(cfg.geometry.totalAngles);
    cfg.geometry.volW = acq.value(QStringLiteral("IMAGE_WIDTH")).toInt(cfg.geometry.volW);
    cfg.geometry.volH = acq.value(QStringLiteral("IMAGE_HEIGHT")).toInt(cfg.geometry.volH);
    cfg.geometry.angleOffsetDeg = static_cast<float>(acq.value(QStringLiteral("BEGINNING_ANGLE_DEG")).toDouble(cfg.geometry.angleOffsetDeg));
    if (!job.value(QStringLiteral("GEOMETRY")).toObject().contains(QStringLiteral("COR_SHIFT")))
    {
        cfg.geometry.corShift = static_cast<float>(acq.value(QStringLiteral("COR_SHIFT")).toDouble(cfg.geometry.corShift));
    }
    cfg.phase.enable = false;
    if (cfg.io.reconMethod == ReconMethod::RealTimeFBP)
    {
        cfg.io.reconMethod = ReconMethod::OfflineFBP;
    }
    if (cfg.geometry.totalAngles <= 0 || cfg.geometry.volW <= 0 || cfg.geometry.volH <= 0)
    {
        const QString message = QStringLiteral("Manifest geometry or angle count is invalid.");
        const QString resultPath = writeReconResultJson(jobPath, job, QStringLiteral("failed"), QStringLiteral("ManifestValidate"),
            QStringLiteral("InvalidGeometry"), QStringLiteral("Manifest geometry or angle count is invalid."),
            projectionDir, cfg.io.saveDir, {}, QDateTime::currentDateTime().toString(Qt::ISODate),
            cfg.geometry.corShift, 0.0, false, cfg.geometry.corShift);
        moveJobToFailed(jobPath, job, resultPath, QStringLiteral("ManifestValidate"),
            QStringLiteral("InvalidGeometry"), message, reconQueueRoot());
        return false;
    }
    const QString firstTomo = QDir(projectionDir).filePath(numberedFileName(cfg.io.tomoPrefix,
        cfg.io.inputPadding, 0, cfg.io.inputExt));
    const QString lastTomo = QDir(projectionDir).filePath(numberedFileName(cfg.io.tomoPrefix,
        cfg.io.inputPadding, cfg.geometry.totalAngles - 1, cfg.io.inputExt));
    if (!QFileInfo::exists(firstTomo) || !QFileInfo::exists(lastTomo))
    {
        const QString message = QStringLiteral("Projection files are incomplete. Missing first or last Tomo file.");
        const QString resultPath = writeReconResultJson(jobPath, job, QStringLiteral("failed"), QStringLiteral("InputScan"),
            QStringLiteral("ProjectionMissing"), message, projectionDir, cfg.io.saveDir, {},
            QDateTime::currentDateTime().toString(Qt::ISODate),
            cfg.geometry.corShift, 0.0, false, cfg.geometry.corShift);
        moveJobToFailed(jobPath, job, resultPath, QStringLiteral("InputScan"),
            QStringLiteral("ProjectionMissing"), message, reconQueueRoot());
        return false;
    }
    if (cfg.io.correctionMode == CorrectionMode::Flat || cfg.io.correctionMode == CorrectionMode::FlatDark)
    {
        const QString firstFlat = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
            cfg.io.inputPadding, 0, cfg.io.inputExt));
        if (!QFileInfo::exists(firstFlat))
        {
            const QString message = QStringLiteral("Flat correction requested but no first Flat file was found.");
            const QString resultPath = writeReconResultJson(jobPath, job, QStringLiteral("failed"), QStringLiteral("ReferenceLoad"),
                QStringLiteral("FlatMissing"), message, projectionDir, cfg.io.saveDir, {},
                QDateTime::currentDateTime().toString(Qt::ISODate),
                cfg.geometry.corShift, 0.0, false, cfg.geometry.corShift);
            moveJobToFailed(jobPath, job, resultPath, QStringLiteral("ReferenceLoad"),
                QStringLiteral("FlatMissing"), message, reconQueueRoot());
            return false;
        }
    }
    if (cfg.geometry.corAuto)
    {
        const QString ref0 = QDir(projectionDir).filePath(cfg.io.filePattern0);
        const QString ref180 = QDir(projectionDir).filePath(cfg.io.filePattern180);
        if (!QFileInfo::exists(ref0) || !QFileInfo::exists(ref180))
        {
            const QString message = QStringLiteral("Auto COR requested but 0/180 reference files are missing.");
            const QString resultPath = writeReconResultJson(jobPath, job, QStringLiteral("failed"), QStringLiteral("ReferenceLoad"),
                QStringLiteral("AutoCorReferenceMissing"), message, projectionDir, cfg.io.saveDir, {},
                QDateTime::currentDateTime().toString(Qt::ISODate),
                cfg.geometry.corShift, 0.0, false, cfg.geometry.corShift);
            moveJobToFailed(jobPath, job, resultPath, QStringLiteral("ReferenceLoad"),
                QStringLiteral("AutoCorReferenceMissing"), message, reconQueueRoot());
            return false;
        }
    }

    const QString runningDir = QDir(reconQueueRoot()).filePath(QStringLiteral("running"));
    QDir().mkpath(runningDir);
    const QString runningPath = QDir(runningDir).filePath(QFileInfo(jobPath).fileName());
    QFile::remove(runningPath);
    job["STATUS"] = "running";
    job["STARTED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    job["PROJECTION_DIR"] = projectionDir;
    job["OUTPUT_DIR"] = cfg.io.saveDir;
    QJsonObject normalizedParams = appConfigToJobObject(cfg);
    for (auto it = normalizedParams.begin(); it != normalizedParams.end(); ++it)
    {
        job[it.key()] = it.value();
    }
    writeJsonFileAtomic(jobPath, job);
    if (!QFile::rename(jobPath, runningPath))
    {
        appendLog(QStringLiteral("Failed to move recon job to running: %1").arg(jobPath));
        return false;
    }
    activeQueueJobPath_ = runningPath;
    activeQueueProjectionDir_ = projectionDir;
    activeQueueOutputDir_ = cfg.io.saveDir;
    activeQueueStartedAt_ = job.value(QStringLiteral("STARTED_AT")).toString();
    activeQueueFinishedStatusOverride_.clear();
    activeQueueFinishedStageOverride_.clear();
    activeQueueFinishedCodeOverride_.clear();
    activeQueueInitialCorShift_ = cfg.geometry.corShift;
    activeQueueAutoCorShift_ = 0.0;
    activeQueueHasAutoCorShift_ = false;
    applyConfig(cfg);
    appendLog(QStringLiteral("Starting queued reconstruction: %1").arg(runningPath));
    refreshReconQueue();
    controller_.startReconstruction(cfg);
    return true;
}

void MainWindow::updateActiveQueueCor(double shift)
{
    if (activeQueueJobPath_.isEmpty())
    {
        return;
    }
    activeQueueAutoCorShift_ = shift;
    activeQueueHasAutoCorShift_ = true;

    QFile jobFile(activeQueueJobPath_);
    if (!jobFile.open(QIODevice::ReadOnly))
    {
        return;
    }
    QJsonObject job = QJsonDocument::fromJson(jobFile.readAll()).object();
    jobFile.close();
    const QString projectionDir = projectionDirFromJob(job).isEmpty() ? activeQueueProjectionDir_ : projectionDirFromJob(job);
    const QString manifestPath = manifestPathForProjectionDir(projectionDir);

    QFile manifestFile(manifestPath);
    if (!manifestFile.open(QIODevice::ReadOnly))
    {
        return;
    }
    QJsonObject manifest = QJsonDocument::fromJson(manifestFile.readAll()).object();
    manifestFile.close();
    QJsonObject acq = manifest.value(QStringLiteral("ACQUISITION")).toObject();
    acq["COR_SHIFT"] = shift;
    acq["COR_UPDATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    manifest["ACQUISITION"] = acq;
    if (writeJsonFileAtomic(manifestPath, manifest))
    {
        appendLog(QStringLiteral("Updated CT manifest COR_SHIFT=%1: %2").arg(shift, 0, 'f', 4).arg(manifestPath));
    }

    QJsonObject jobGeometry = job.value(QStringLiteral("GEOMETRY")).toObject();
    jobGeometry["COR_SHIFT"] = shift;
    jobGeometry["COR_AUTO"] = false;
    job["GEOMETRY"] = jobGeometry;
    writeJsonFileAtomic(activeQueueJobPath_, job);

    saveSessionConfig();
}

QString MainWindow::writeReconResultJson(const QString& jobPath, const QJsonObject& job, const QString& status,
    const QString& failedStage, const QString& errorCode, const QString& message,
    const QString& projectionDir, const QString& outputDir, const QString& startedAt,
    const QString& finishedAt, double initialCor, double autoCor, bool hasAutoCor, double finalCor)
{
    QJsonObject result;
    result["TYPE"] = "CT_STREAM_RECON_RESULT";
    result["VERSION"] = 1;
    result["STATUS"] = status;
    result["FAILED_STAGE"] = failedStage;
    result["ERROR_CODE"] = errorCode;
    result["ERROR_MESSAGE"] = message;
    result["JOB_PATH"] = jobPath;
    result["PROJECTION_DIR"] = projectionDir;
    result["OUTPUT_DIR"] = outputDir;
    result["STARTED_AT"] = startedAt;
    result["FINISHED_AT"] = finishedAt;
    result["INITIAL_COR_SHIFT"] = initialCor;
    if (hasAutoCor)
    {
        result["AUTO_COR_SHIFT"] = autoCor;
    }
    result["FINAL_COR_SHIFT"] = finalCor;
    result["RECON_METHOD"] = job.value(QStringLiteral("RECON_METHOD")).toString();

    QJsonArray tail;
    const int first = std::max(0, static_cast<int>(recentLogLines_.size()) - 40);
    for (int i = first; i < recentLogLines_.size(); ++i)
    {
        tail.append(recentLogLines_[i]);
    }
    result["LOG_TAIL"] = tail;

    QString resultPath;
    if (!outputDir.isEmpty() && QDir().mkpath(outputDir))
    {
        resultPath = QDir(outputDir).filePath(QStringLiteral("recon_result.json"));
    }
    else
    {
        const QString failedDir = QDir(reconQueueRoot()).filePath(QStringLiteral("failed"));
        QDir().mkpath(failedDir);
        resultPath = QDir(failedDir).filePath(reconResultFileName(jobPath));
    }
    if (!writeJsonFileAtomic(resultPath, result))
    {
        const QString failedDir = QDir(reconQueueRoot()).filePath(QStringLiteral("failed"));
        QDir().mkpath(failedDir);
        resultPath = QDir(failedDir).filePath(reconResultFileName(jobPath));
        writeJsonFileAtomic(resultPath, result);
    }
    return resultPath;
}

void MainWindow::finishQueuedReconstruction(bool ok, const QString& message, const QString& failedStage, const QString& errorCode)
{
    if (activeQueueJobPath_.isEmpty())
    {
        if (ok)
        {
            startNextQueuedReconstruction();
        }
        return;
    }
    QJsonObject job;
    readJsonFile(activeQueueJobPath_, &job);
    const bool cancelled = activeQueueFinishedStatusOverride_ == QStringLiteral("cancelled");
    const QString status = cancelled ? QStringLiteral("cancelled") : (ok ? QStringLiteral("done") : QStringLiteral("failed"));
    const QString stage = cancelled
        ? (activeQueueFinishedStageOverride_.isEmpty() ? QStringLiteral("UserStopped") : activeQueueFinishedStageOverride_)
        : (ok ? QString() : (failedStage.isEmpty() ? QStringLiteral("Reconstruction") : failedStage));
    const QString code = cancelled
        ? (activeQueueFinishedCodeOverride_.isEmpty() ? QStringLiteral("UserStopped") : activeQueueFinishedCodeOverride_)
        : (ok ? QString() : (errorCode.isEmpty() ? QStringLiteral("ReconstructionFailed") : errorCode));
    const QString finishedAt = QDateTime::currentDateTime().toString(Qt::ISODate);
    const double finalCor = activeQueueHasAutoCorShift_ ? activeQueueAutoCorShift_ : activeQueueInitialCorShift_;
    const QString resultPath = writeReconResultJson(activeQueueJobPath_, job, status, stage, code, message,
        activeQueueProjectionDir_, activeQueueOutputDir_, activeQueueStartedAt_, finishedAt,
        activeQueueInitialCorShift_, activeQueueAutoCorShift_, activeQueueHasAutoCorShift_, finalCor);

    const QString bucket = status == QStringLiteral("done") ? QStringLiteral("done") : QStringLiteral("failed");
    const QString targetDir = QDir(reconQueueRoot()).filePath(bucket);
    QDir().mkpath(targetDir);
    const QString targetPath = QDir(targetDir).filePath(QFileInfo(activeQueueJobPath_).fileName());

    QFile file(activeQueueJobPath_);
    if (file.open(QIODevice::ReadOnly))
    {
        job = QJsonDocument::fromJson(file.readAll()).object();
        file.close();
        job["STATUS"] = status;
        job["FINISHED_AT"] = finishedAt;
        job["MESSAGE"] = message;
        job["FAILED_STAGE"] = stage;
        job["ERROR_CODE"] = code;
        job["ERROR_MESSAGE"] = message;
        job["RESULT_PATH"] = resultPath;
        QJsonObject geometry = job.value(QStringLiteral("GEOMETRY")).toObject();
        geometry["COR_SHIFT"] = finalCor;
        geometry["COR_AUTO"] = false;
        job["GEOMETRY"] = geometry;
        writeJsonFileAtomic(activeQueueJobPath_, job);
    }
    QFile::remove(targetPath);
    if (!QFile::rename(activeQueueJobPath_, targetPath))
    {
        appendLog(QStringLiteral("Failed to move recon job to %1: %2").arg(bucket, activeQueueJobPath_));
    }
    activeQueueJobPath_.clear();
    activeQueueProjectionDir_.clear();
    activeQueueOutputDir_.clear();
    activeQueueStartedAt_.clear();
    activeQueueFinishedStatusOverride_.clear();
    activeQueueFinishedStageOverride_.clear();
    activeQueueFinishedCodeOverride_.clear();
    activeQueueInitialCorShift_ = 0.0;
    activeQueueAutoCorShift_ = 0.0;
    activeQueueHasAutoCorShift_ = false;
    refreshReconQueue();
    if (status == QStringLiteral("done"))
    {
        startNextQueuedReconstruction();
    }
}

void MainWindow::stopReconstruction()
{
    controller_.stopReconstruction();
    if (!activeQueueJobPath_.isEmpty())
    {
        activeQueueFinishedStatusOverride_ = QStringLiteral("cancelled");
        activeQueueFinishedStageOverride_ = QStringLiteral("UserStopped");
        activeQueueFinishedCodeOverride_ = QStringLiteral("UserStopped");
        appendLog(QStringLiteral("Queued reconstruction stop requested; waiting for worker to exit."));
    }
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
        if (ctRunning_ && !ok && label.startsWith(QStringLiteral("CT ")))
        {
            finishCtAcquisition(false, QStringLiteral("%1 failed: %2").arg(label, response));
        }
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
    const bool firstPosition = !axisUi.hasPosition;
    const double userValue = static_cast<double>(positionPulse) / axisUi.pulsePerUnit;
    axisUi.currentValue = userValue;
    axisUi.hasPosition = true;
    axisUi.positionLabel->setText(QStringLiteral("%1").arg(userValue, 0, 'f', 4));
    axisUi.positionLabel->setToolTip(QStringLiteral("%1 pulse\n%2").arg(positionPulse).arg(statusRaw));
    if (firstPosition && axisUi.absoluteSpin)
    {
        const QSignalBlocker blocker(axisUi.absoluteSpin);
        axisUi.absoluteSpin->setValue(userValue);
    }
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
    axisUi.goButton->setEnabled(motorConnected_ && !active);
    axisUi.relativeMinusButton->setEnabled(motorConnected_ && !active && !motorSpeedMode_);
    axisUi.relativePlusButton->setEnabled(motorConnected_ && !active && !motorSpeedMode_);
    if (ctRunning_ && ctWaitingForMotor_ && !active && (axis == ctSettings_.rAxis || axis == ctSettings_.yAxis))
    {
        const QString action = ctPendingMotorAction_;
        ctPendingMotorAction_.clear();
        ctWaitingForMotor_ = false;
        if (action == QStringLiteral("flat_out"))
        {
            ctYIsOutForFlat_ = true;
        }
        else if (action == QStringLiteral("flat_return"))
        {
            ctYIsOutForFlat_ = false;
        }
        QTimer::singleShot(0, this, &MainWindow::advanceCtAcquisition);
    }
}

void MainWindow::moveMotorAxis(const QString& name)
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

    if (!it->hasPosition)
    {
        appendLog(QStringLiteral("%1 current position is not available yet; wait for position polling before moving.").arg(name));
        return;
    }

    const double absolute = it->absoluteSpin->value();
    const double target = absolute;
    if (target < it->softMin || target > it->softMax)
    {
        it->absoluteSpin->setValue(it->currentValue);
        appendLog(QStringLiteral("%1 move exceeds UI soft limit: target %2 %3, allowed [%4, %5] %3. Input reset.")
            .arg(name)
            .arg(target, 0, 'f', 4)
            .arg(it->unit)
            .arg(it->softMin, 0, 'f', 4)
            .arg(it->softMax, 0, 'f', 4));
        return;
    }

    const qint64 pulse = static_cast<qint64>(std::llround(target * it->pulsePerUnit));
    if (pulse == 0)
    {
        appendLog(QStringLiteral("%1 absolute move is too small for pulse conversion.").arg(name));
        return;
    }

    const QString label = QStringLiteral("%1 absolute %2 pulse (%3 -> %4 %5)")
        .arg(name)
        .arg(pulse)
        .arg(it->currentValue, 0, 'f', 4)
        .arg(target, 0, 'f', 4)
        .arg(it->unit);
    it->active = true;
    it->goButton->setEnabled(false);
    QMetaObject::invokeMethod(
        kohzuController_,
        "moveAbsolutePulse",
        Qt::QueuedConnection,
        Q_ARG(int, it->axis),
        Q_ARG(qint64, pulse),
        Q_ARG(int, 0),
        Q_ARG(QString, label));
}

void MainWindow::moveMotorAxisRelative(const QString& name, int direction)
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
    if (!it->hasPosition)
    {
        appendLog(QStringLiteral("%1 current position is not available yet; wait for position polling before moving.").arg(name));
        return;
    }

    const double step = std::abs(it->relativeSpin->value());
    if (step <= 0.0)
    {
        appendLog(QStringLiteral("%1 relative move is zero.").arg(name));
        return;
    }
    const double relative = step * (direction < 0 ? -1.0 : 1.0);
    const double target = it->currentValue + relative;
    if (target < it->softMin || target > it->softMax)
    {
        it->relativeSpin->setValue(0.0);
        appendLog(QStringLiteral("%1 relative move exceeds UI soft limit: target %2 %3, allowed [%4, %5] %3. Input reset.")
            .arg(name)
            .arg(target, 0, 'f', 4)
            .arg(it->unit)
            .arg(it->softMin, 0, 'f', 4)
            .arg(it->softMax, 0, 'f', 4));
        return;
    }

    const qint64 pulse = static_cast<qint64>(std::llround(relative * it->pulsePerUnit));
    if (pulse == 0)
    {
        appendLog(QStringLiteral("%1 relative move is too small for pulse conversion.").arg(name));
        return;
    }

    const QString label = QStringLiteral("%1 relative %2 pulse (%3 -> %4 %5)")
        .arg(name)
        .arg(pulse)
        .arg(it->currentValue, 0, 'f', 4)
        .arg(target, 0, 'f', 4)
        .arg(it->unit);
    it->active = true;
    it->relativeMinusButton->setEnabled(false);
    it->relativePlusButton->setEnabled(false);
    it->goButton->setEnabled(false);
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

void MainWindow::stopAllMotorAxes()
{
    if (!motorConnected_)
    {
        appendLog(QStringLiteral("Kohzu is not connected."));
        return;
    }
    appendLog(QStringLiteral("Stopping all Kohzu axes with STP0/0 deceleration stop."));
    QMetaObject::invokeMethod(kohzuController_, "stopAllAxes", Qt::QueuedConnection);
}

void MainWindow::toggleMotorSpeedMode()
{
    motorSpeedMode_ = !motorSpeedMode_;
    motorModeButton_->setText(motorSpeedMode_ ? QStringLiteral("Motion") : QStringLiteral("Speed"));
    motorMotionHeader_->setVisible(!motorSpeedMode_);
    motorSpeedHeader_->setVisible(motorSpeedMode_);
    for (auto it = motorAxes_.begin(); it != motorAxes_.end(); ++it)
    {
        it->positionLabel->setVisible(!motorSpeedMode_);
        it->unitLabel->setVisible(!motorSpeedMode_);
        it->relativeWidget->setVisible(!motorSpeedMode_);
        it->relativeMinusButton->setVisible(!motorSpeedMode_);
        it->relativeSpin->setVisible(!motorSpeedMode_);
        it->relativePlusButton->setVisible(!motorSpeedMode_);
        it->absoluteSpin->setVisible(!motorSpeedMode_);
        it->goButton->setVisible(!motorSpeedMode_);
        it->speedUnitLabel->setVisible(motorSpeedMode_);
        it->speedSpin->setVisible(motorSpeedMode_);
        it->accelSpin->setVisible(motorSpeedMode_);
    }
    updateMotorUiEnabled();
}

void MainWindow::updateMotorUiEnabled()
{
    motorHostEdit_->setEnabled(!motorConnected_);
    motorPortSpin_->setEnabled(!motorConnected_);
    motorModeButton_->setEnabled(true);
    motorStopAllButton_->setEnabled(motorConnected_);
    for (auto it = motorAxes_.begin(); it != motorAxes_.end(); ++it)
    {
        it->goButton->setEnabled(motorConnected_ && !it->active);
        it->relativeMinusButton->setEnabled(motorConnected_ && !it->active && !motorSpeedMode_);
        it->relativePlusButton->setEnabled(motorConnected_ && !it->active && !motorSpeedMode_);
        it->relativeSpin->setEnabled(!it->active && !motorSpeedMode_);
        it->absoluteSpin->setEnabled(!it->active && !motorSpeedMode_);
        it->speedSpin->setEnabled(motorSpeedMode_);
        it->accelSpin->setEnabled(motorSpeedMode_);
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

    if (ctWriter_)
    {
        disconnect(ctWriter_, nullptr, this, nullptr);
        CtTiffWriter* writer = ctWriter_;
        QThread* guiThread = thread();
        QMetaObject::invokeMethod(writer, [writer]() {
            writer->stop();
        }, Qt::BlockingQueuedConnection);
        QMetaObject::invokeMethod(writer, [writer, guiThread]() {
            writer->moveToThread(guiThread);
        }, Qt::BlockingQueuedConnection);
        ctWriter_ = nullptr;
        ctWriterThread_.quit();
        ctWriterThread_.wait(3000);
        delete writer;
    }
    else
    {
        ctWriterThread_.quit();
        ctWriterThread_.wait(3000);
    }

    if (cameraWorker_)
    {
        disconnect(cameraWorker_, nullptr, this, nullptr);
        PvcamCameraWorker* worker = cameraWorker_;
        QThread* guiThread = thread();
        worker->stopCapture();
        QMetaObject::invokeMethod(worker, [worker]() {
            worker->shutdown();
        }, Qt::BlockingQueuedConnection);
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
