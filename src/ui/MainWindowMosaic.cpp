#include "core/io/ImageFileIo.h"
#include "ui/MainWindow.h"
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSaveFile>
#include <QScrollArea>
#include <QSpinBox>
#include <QStorageInfo>
#include <QTabWidget>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ct
{
namespace
{
bool atomicJson(const QString &path, const QJsonObject &o)
{
    QSaveFile f(path);
    if (!f.open(QIODevice::WriteOnly))
        return false;
    auto b = QJsonDocument(o).toJson();
    return f.write(b) == b.size() && f.commit();
}
double estimatedMosaicBytes(const MosaicPlan &p, int angles, int flats)
{
    const double count = double(p.tiles.size());
    const double raw = count * p.width * p.height * 2. * (angles + 2. * flats + 2);
    const double tiles = count * p.width * p.width * p.height * 4.;
    const auto &c = p.config;
    const double angle = c.motorToReconDeg * std::acos(-1.) / 180;
    const double x = std::abs(std::cos(angle)) * c.extentX0Um + std::abs(std::sin(angle)) * c.extentX90Um;
    const double y = std::abs(std::sin(angle)) * c.extentX0Um + std::abs(std::cos(angle)) * c.extentX90Um;
    const double fused = x * y * c.extentZUm / std::pow(c.pixelUm, 3) * 4;
    const double temporary = double(angles) * p.width * p.height * 4;
    return (raw + tiles + fused + temporary) * 1.1;
}
} // namespace
QWidget *MainWindow::createMosaicPage()
{
    auto *page = new QWidget(ctModeTabs_);
    auto *form = new QFormLayout(page);
    auto real = [&](double low, double high, int decimals, double value) {
        auto *s = new QDoubleSpinBox(page);
        s->setRange(low, high);
        s->setDecimals(decimals);
        s->setValue(value);
        s->setButtonSymbols(QAbstractSpinBox::NoButtons);
        connect(s, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this]() { updateMosaicPlan(); });
        return s;
    };
    mosaicPixel_ = real(0.001, 100, 4, 0.65);
    mosaicX0_ = real(0.001, 100, 3, 10);
    mosaicX90_ = real(0.001, 100, 3, 1);
    mosaicZ_ = real(0.001, 100, 3, 8);
    mosaicOverlap_ = real(10, 80, 1, 25);
    mosaicDiameter_ = real(50, 98, 1, 90);
    mosaicAngle_ = real(-360, 360, 4, 0);
    mosaicSettle_ = new QSpinBox(page);
    mosaicSettle_->setRange(0, 60000);
    mosaicSettle_->setValue(300);
    mosaicSettle_->setButtonSymbols(QAbstractSpinBox::NoButtons);
    mosaicX90Sign_ = new QComboBox(page);
    mosaicX90Sign_->addItems({"+1", "-1"});
    mosaicZSign_ = new QComboBox(page);
    mosaicZSign_->addItems({"-1", "+1"});
    mosaicCalibrated_ = new QCheckBox("Axis mapping calibrated", page);
    mosaicFlatMove_ = real(-150000, 150000, 2, 0);
    mosaicFlatCount_ = new QSpinBox(page);
    mosaicFlatCount_->setRange(0, 10000);
    mosaicFlatCount_->setValue(10);
    mosaicFlatCount_->setButtonSymbols(QAbstractSpinBox::NoButtons);
    form->addRow(new QLabel("Local step CT: X0 / X90 / Z", page));
    form->addRow("Pixel um", mosaicPixel_);
    form->addRow("Sample X0 mm", mosaicX0_);
    form->addRow("Sample X90 mm", mosaicX90_);
    form->addRow("Sample Z mm", mosaicZ_);
    form->addRow("Overlap %", mosaicOverlap_);
    form->addRow("Valid diameter %", mosaicDiameter_);
    form->addRow("Settle ms", mosaicSettle_);
    form->addRow("Flat move Y um", mosaicFlatMove_);
    form->addRow("Flats per group", mosaicFlatCount_);
    mosaicDarkPath_ = new QLineEdit(page);
    mosaicDarkConfirmed_ = new QCheckBox("Dark matches exposure, ROI and readout", page);
    connect(mosaicDarkPath_, &QLineEdit::textChanged, this, [this]() { mosaicDarkConfirmed_->setChecked(false); });
    auto *darkBrowse = new QPushButton("Select measured dark TIFF", page);
    connect(darkBrowse, &QPushButton::clicked, this, [this]() {
        auto path = QFileDialog::getOpenFileName(this, "Measured dark (single frame or averaged image)", {},
                                                 "TIFF (*.tif *.tiff)");
        if (!path.isEmpty())
        {
            mosaicDarkPath_->setText(path);
            mosaicDarkConfirmed_->setChecked(false);
        }
    });
    form->addRow(darkBrowse);
    form->addRow("Dark reference", mosaicDarkPath_);
    form->addRow(mosaicDarkConfirmed_);
    form->addRow("X0 to recon XY deg", mosaicAngle_);
    form->addRow("X90 handedness", mosaicX90Sign_);
    form->addRow("Z image direction", mosaicZSign_);
    form->addRow(mosaicCalibrated_);
    mosaicPlanLabel_ = new QLabel(page);
    mosaicPlanLabel_->setWordWrap(true);
    form->addRow(mosaicPlanLabel_);
    auto *note = new QLabel("Align sample centre before Start. A new tile is acquired only after R, X0, X90 and Z "
                            "positioning completes. Uses Step flats and 180-degree projections. No automatic dark "
                            "shutter is configured; dark correction requires measured dark data.",
                            page);
    note->setWordWrap(true);
    form->addRow(note);
    mosaicManifestEdit_ = new QLineEdit(page);
    auto *browse = new QPushButton("Open mosaic manifest", page);
    connect(browse, &QPushButton::clicked, this, [this]() {
        auto p = QFileDialog::getOpenFileName(this, "Mosaic acquisition manifest", {}, "JSON (*.json)");
        if (!p.isEmpty())
            mosaicManifestEdit_->setText(p);
    });
    form->addRow(browse);
    form->addRow("Mosaic manifest", mosaicManifestEdit_);
    mosaicTileReconName_ = new QLineEdit("Recon", page);
    mosaicOutputName_ = new QLineEdit("MosaicRecon", page);
    mosaicRegistration_ = new QLineEdit(page);
    form->addRow("Tile recon folder", mosaicTileReconName_);
    form->addRow("Output folder", mosaicOutputName_);
    form->addRow("Reuse registration JSON", mosaicRegistration_);
    mosaicStitchButton_ = new QPushButton("Register and stitch volumes", page);
    mosaicTestButton_ = new QPushButton("Synthetic pipeline test (no motion)", page);
    connect(mosaicStitchButton_, &QPushButton::clicked, this, [this]() { launchMosaicStitch(false); });
    connect(mosaicTestButton_, &QPushButton::clicked, this, [this]() { launchMosaicStitch(true); });
    form->addRow(mosaicStitchButton_);
    form->addRow(mosaicTestButton_);
    auto *requeue = new QPushButton("Reconstruct all tiles with current parameters", page);
    connect(requeue, &QPushButton::clicked, this, &MainWindow::queueMosaicReconstruction);
    form->addRow(requeue);
    auto *stop = new QPushButton("Stop stitching", page);
    connect(stop, &QPushButton::clicked, this, [this]() {
        mosaicAutoFusionManifest_.clear();
        if (mosaicProcess_ && mosaicProcess_->state() != QProcess::NotRunning)
            mosaicProcess_->write("stop\n");
    });
    form->addRow(stop);
    connect(mosaicCalibrated_, &QCheckBox::toggled, this, [this]() { updateMosaicPlan(); });
    connect(mosaicAngle_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this]() {
        if (!applyingConfig_)
            mosaicCalibrated_->setChecked(false);
    });
    for (auto *combo : {mosaicX90Sign_, mosaicZSign_})
        connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this]() {
            if (!applyingConfig_)
                mosaicCalibrated_->setChecked(false);
        });
    for (auto *s : {cameraRoiWidthSpin_, cameraRoiHeightSpin_})
        connect(s, qOverload<int>(&QSpinBox::valueChanged), this, [this]() { updateMosaicPlan(); });
    for (auto *s : {cameraRoiWidthSpin_, cameraRoiHeightSpin_, cameraRoiXSpin_, cameraRoiYSpin_})
        connect(s, qOverload<int>(&QSpinBox::valueChanged), this,
                [this]() { mosaicDarkConfirmed_->setChecked(false); });
    connect(cameraExposureSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this]() { mosaicDarkConfirmed_->setChecked(false); });
    connect(cameraExposureUnitCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this]() { mosaicDarkConfirmed_->setChecked(false); });
    connect(ctProjectionNumSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this]() { updateMosaicPlan(); });
    connect(mosaicFlatCount_, qOverload<int>(&QSpinBox::valueChanged), this, [this]() { updateMosaicPlan(); });
    auto *scroll = new QScrollArea(ctModeTabs_);
    scroll->setObjectName("mosaicPage");
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidget(page);
    scroll->setMinimumHeight(200);
    scroll->setMaximumHeight(400);
    return scroll;
}
MosaicConfig MainWindow::mosaicConfigFromUi() const
{
    MosaicConfig c;
    if (!mosaicPixel_)
        return c;
    c.pixelUm = mosaicPixel_->value();
    c.extentX0Um = mosaicX0_->value() * 1000;
    c.extentX90Um = mosaicX90_->value() * 1000;
    c.extentZUm = mosaicZ_->value() * 1000;
    c.overlap = mosaicOverlap_->value() / 100;
    c.usableDiameter = mosaicDiameter_->value() / 100;
    c.settleMs = mosaicSettle_->value();
    c.flatMoveUm = mosaicFlatMove_->value();
    c.flatCount = mosaicFlatCount_->value();
    c.darkPath = mosaicDarkPath_->text();
    c.calibrated = mosaicCalibrated_->isChecked();
    c.motorToReconDeg = mosaicAngle_->value();
    c.x90Sign = mosaicX90Sign_->currentText().toInt();
    c.zSign = mosaicZSign_->currentText().toInt();
    return c;
}
void MainWindow::applyMosaicConfig(const MosaicConfig &c)
{
    mosaicPixel_->setValue(c.pixelUm);
    mosaicX0_->setValue(c.extentX0Um / 1000);
    mosaicX90_->setValue(c.extentX90Um / 1000);
    mosaicZ_->setValue(c.extentZUm / 1000);
    mosaicOverlap_->setValue(c.overlap * 100);
    mosaicDiameter_->setValue(c.usableDiameter * 100);
    mosaicSettle_->setValue(c.settleMs);
    mosaicFlatMove_->setValue(c.flatMoveUm);
    mosaicFlatCount_->setValue(c.flatCount);
    mosaicDarkPath_->setText(c.darkPath);
    mosaicDarkConfirmed_->setChecked(false);
    mosaicAngle_->setValue(c.motorToReconDeg);
    mosaicX90Sign_->setCurrentIndex(c.x90Sign == 1 ? 0 : 1);
    mosaicZSign_->setCurrentIndex(c.zSign == -1 ? 0 : 1);
    mosaicCalibrated_->setChecked(c.calibrated);
    updateMosaicPlan();
}
void MainWindow::updateMosaicPlan()
{
    if (!mosaicPlanLabel_ || !mosaicCalibrated_)
        return;
    try
    {
        auto p = makeMosaicPlan(mosaicConfigFromUi(), cameraRoiWidthSpin_->value(), cameraRoiHeightSpin_->value());
        mosaicPlanLabel_->setText(
            QString("%1 tiles; valid core %2 x %3 mm. Storage estimate %4 TiB (raw, float tiles, fusion, temporary, "
                    "10% margin). %5")
                .arg(p.tiles.size())
                .arg(p.coreXUm / 1000, 0, 'f', 3)
                .arg(p.coreYUm / 1000, 0, 'f', 3)
                .arg(estimatedMosaicBytes(p, ctProjectionNumSpin_->value(), mosaicFlatCount_->value()) /
                         std::pow(1024., 4),
                     0, 'f', 2)
                .arg(p.config.calibrated ? "Axis mapping confirmed."
                                         : "Calibrate mapping before hardware acquisition."));
    }
    catch (const std::exception &e)
    {
        mosaicPlanLabel_->setText(QString::fromUtf8(e.what()));
    }
}
void MainWindow::setMosaicLocked(bool locked)
{
    for (auto *w : {static_cast<QWidget *>(ctModeTabs_), static_cast<QWidget *>(ctSavePathEdit_),
                    static_cast<QWidget *>(ctProjectionNumSpin_), static_cast<QWidget *>(ctBeginningAngleSpin_),
                    static_cast<QWidget *>(cameraExposureSpin_), static_cast<QWidget *>(cameraExposureUnitCombo_),
                    static_cast<QWidget *>(cameraRoiWidthSpin_), static_cast<QWidget *>(cameraRoiHeightSpin_),
                    static_cast<QWidget *>(cameraApplyRoiButton_), static_cast<QWidget *>(cameraResetRoiButton_),
                    static_cast<QWidget *>(cameraRoiXSpin_), static_cast<QWidget *>(cameraRoiYSpin_)})
        w->setEnabled(!locked);
    ctStartButton_->setEnabled(!locked && !ctHardwareStopPending_ && !ctRecoveryPending_);
    ctStopButton_->setEnabled(locked && (mosaicRunning_ || ctRunning_));
    cameraSnapButton_->setEnabled(!locked);
    cameraLiveButton_->setEnabled(!locked);
    for (auto it = motorAxes_.begin(); it != motorAxes_.end(); ++it)
    {
        const bool ready = !locked && motorConnected_ && !it->active && !ctHardwareStopPending_;
        it->goButton->setEnabled(ready);
        it->relativeMinusButton->setEnabled(ready && !motorSpeedMode_);
        it->relativePlusButton->setEnabled(ready && !motorSpeedMode_);
        it->baseSpeedSpin->setEnabled(!locked && motorSpeedMode_);
        it->speedSpin->setEnabled(!locked && motorSpeedMode_);
        it->accelSpin->setEnabled(!locked && motorSpeedMode_);
    }
}
void MainWindow::startMosaicAcquisition()
{
    if (mosaicRunning_ || ctRunning_ || timingCalibrationRunning_ ||
        (mosaicProcess_ && mosaicProcess_->state() != QProcess::NotRunning))
    {
        appendLog("Mosaic cannot start while acquisition is busy.");
        return;
    }
    try
    {
        if (!motorConnected_ || !cameraWorker_ || !ctWriter_ || ctHardwareStopPending_ ||
            (ctRecoveryPending_ && !ctRecoveryConfirmed_))
            throw std::runtime_error("Hardware is not ready or recovery is pending.");
        mosaicPlan_ = makeMosaicPlan(mosaicConfigFromUi(), cameraRoiWidthSpin_->value(), cameraRoiHeightSpin_->value());
        if (!mosaicPlan_.config.calibrated)
            throw std::runtime_error("Calibrate the X0/X90/Z to reconstruction coordinate mapping first.");
        mosaicRoot_ = QDir::cleanPath(ctSavePathEdit_->text().trimmed());
        if (ctSavePathEdit_->text().trimmed().isEmpty() || QFileInfo::exists(mosaicRoot_))
            throw std::runtime_error("Mosaic Save Path must be a new directory.");
        QString existing = QFileInfo(mosaicRoot_).absolutePath();
        while (!QFileInfo::exists(existing))
        {
            QString parent = QFileInfo(existing).absolutePath();
            if (parent == existing)
                break;
            existing = parent;
        }
        QStorageInfo storage(existing);
        const double required =
            estimatedMosaicBytes(mosaicPlan_, ctProjectionNumSpin_->value(), mosaicFlatCount_->value());
        if (storage.isValid() && storage.isReady() && storage.bytesAvailable() >= 0 &&
            storage.bytesAvailable() < required)
            throw std::runtime_error(QString("Insufficient storage: estimated %1 TiB, available %2 TiB. Reduce "
                                             "extent/ROI or select a larger volume.")
                                         .arg(required / std::pow(1024., 4), 0, 'f', 2)
                                         .arg(storage.bytesAvailable() / std::pow(1024., 4), 0, 'f', 2)
                                         .toStdString());
        for (const QString name : {"R", "Y", "X0", "X90", "Z"})
            if (!motorAxes_.contains(name) || !motorAxes_[name].hasPosition || motorAxes_[name].active)
                throw std::runtime_error("All R/Y/X0/X90/Z positions must be known and idle.");
        const QStringList axes{"X0", "X90", "Z"};
        for (int j = 0; j < 3; ++j)
            mosaicOrigin_[j] = motorAxes_[axes[j]].currentValue;
        for (auto &t : mosaicPlan_.tiles)
            for (int j = 0; j < 3; ++j)
            {
                double target = mosaicOrigin_[j] + t.motorOffset[j];
                const auto &a = motorAxes_[axes[j]];
                if (target < a.softMin || target > a.softMax)
                    throw std::runtime_error(QString("Tile %1 exceeds %2 travel; recenter the sample or reduce extent.")
                                                 .arg(t.id)
                                                 .arg(axes[j])
                                                 .toStdString());
            }
        auto &r = motorAxes_["R"];
        if (ctBeginningAngleSpin_->value() < r.softMin || ctBeginningAngleSpin_->value() + 180 > r.softMax)
            throw std::runtime_error("R scan range exceeds limits.");
        auto &y = motorAxes_["Y"];
        mosaicOriginY_ = y.currentValue;
        double flat = y.currentValue + mosaicFlatMove_->value();
        if (mosaicFlatCount_->value() > 0 && (flat < y.softMin || flat > y.softMax || mosaicFlatMove_->value() == 0))
            throw std::runtime_error("Configure a non-zero, in-range flat Y movement.");
        mosaicDarkSnapshot_ = mosaicDarkPath_->text().trimmed();
        if (!mosaicDarkSnapshot_.isEmpty())
        {
            auto dark = readImageFile(mosaicDarkSnapshot_);
            if (dark.empty() || dark.channels() != 1 || dark.cols != mosaicPlan_.width ||
                dark.rows != mosaicPlan_.height || !mosaicDarkConfirmed_->isChecked())
                throw std::runtime_error("Dark reference dimensions or exposure/ROI confirmation is missing.");
        }
        else if (configFromUi().io.correctionMode == CorrectionMode::FlatDark)
            throw std::runtime_error("Flat-Dark correction requires a measured matching dark TIFF.");
        mosaicSnapshot_ = configFromUi();
        mosaicSnapshot_.io.saveDepth = SaveDepth::Bit32;
        mosaicSnapshot_.io.saveFormat = "tiff";
        if (mosaicFlatCount_->value() == 0 && mosaicSnapshot_.io.correctionMode != CorrectionMode::None)
            throw std::runtime_error("Flat correction requires Flats per group greater than zero.");
        if (!mosaicDarkSnapshot_.isEmpty())
            mosaicSnapshot_.io.correctionMode = CorrectionMode::FlatDark;
        mosaicSnapshot_.phase.pixelSizeUm = float(mosaicPlan_.config.pixelUm);
        if (mosaicSnapshot_.io.reconMethod == ReconMethod::RealTimeFBP)
            mosaicSnapshot_.io.reconMethod = ReconMethod::OfflineFBP;
        mosaicSnapshot_.ctAcquisitionUi.stepFlatGap = 0;
        mosaicSnapshot_.ctAcquisitionUi.stepFlatNum = mosaicFlatCount_->value();
        mosaicSnapshot_.ctAcquisitionUi.stepFlatMoveUm = mosaicFlatMove_->value();
        mosaicTileRecords_.clear();
        for (auto &t : mosaicPlan_.tiles)
            mosaicTileRecords_.append(t.toJson());
        mosaicRecord_ = mosaicPlan_.toJson();
        mosaicRecord_["CREATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
        mosaicRecord_["SOFTWARE_VERSION"] = QStringLiteral(CT_STREAM_VERSION);
        mosaicRecord_["CALIBRATED_RECON_METHOD"] = toString(mosaicSnapshot_.io.reconMethod);
        mosaicRecord_["ORIGIN_CONTROLLER_UM"] = QJsonArray{mosaicOrigin_[0], mosaicOrigin_[1], mosaicOrigin_[2]};
        if (!QDir().mkpath(mosaicRoot_))
            throw std::runtime_error("Cannot create mosaic directory.");
        if (!mosaicDarkSnapshot_.isEmpty())
        {
            QString destination = QDir(mosaicRoot_).filePath("measured_dark.tif");
            if (!QFile::copy(mosaicDarkSnapshot_, destination))
                throw std::runtime_error("Could not snapshot dark reference.");
            mosaicDarkSnapshot_ = destination;
            QFile file(destination);
            if (!file.open(QIODevice::ReadOnly))
                throw std::runtime_error("Cannot read dark snapshot for checksum.");
            mosaicRecord_["DARK_SHA256"] =
                QString::fromLatin1(QCryptographicHash::hash(file.readAll(), QCryptographicHash::Sha256).toHex());
            mosaicRecord_["DARK_REFERENCE"] = "measured_dark.tif";
            mosaicRecord_["DARK_MATCH_CONFIRMED"] = true;
        }
        mosaicAutoFusionManifest_ = QDir(mosaicRoot_).filePath("mosaic_manifest.json");
        mosaicAutoFusionOutput_ = "MosaicRecon";
        mosaicAutoFusionTileRecon_ = "Recon";
        mosaicAutoFusionRegistration_.clear();
        mosaicRunning_ = true;
        ++mosaicGeneration_;
        mosaicIndex_ = 0;
        mosaicPositionStage_ = 0;
        mosaicRestoreLive_ = cameraLiveRunning_;
        if (cameraLiveRunning_)
            stopCameraLivePreview(false);
        writeMosaicManifest(false);
        if (!mosaicRunning_)
            return;
        writeCtRecoveryState("RUNNING", "Mosaic positioning started.");
        mosaicManifestEdit_->setText(QDir(mosaicRoot_).filePath("mosaic_manifest.json"));
        setMosaicLocked(true);
        appendLog(
            QString("Mosaic starting: %1 tiles, local step CT, float32 reconstruction.").arg(mosaicPlan_.tiles.size()));
        advanceMosaicPositioning();
    }
    catch (const std::exception &e)
    {
        if (mosaicRunning_)
            abortMosaic(QString::fromUtf8(e.what()));
        else
            appendLog("Mosaic start failed: " + QString::fromUtf8(e.what()));
    }
}
void MainWindow::writeMosaicManifest(bool completed, const QString &error)
{
    QJsonArray a;
    for (auto &t : mosaicTileRecords_)
        a.append(t);
    mosaicRecord_["TILES"] = a;
    mosaicRecord_["COMPLETED"] = completed;
    mosaicRecord_["ERROR"] = error;
    if (!atomicJson(QDir(mosaicRoot_).filePath("mosaic_manifest.json"), mosaicRecord_))
    {
        appendLog("Mosaic manifest write failed; stopping acquisition.");
        mosaicRunning_ = false;
        mosaicAutoFusionManifest_.clear();
        ++mosaicGeneration_;
        mosaicPendingAxis_.clear();
        writeCtRecoveryState("STOPPING", "Mosaic manifest write failed.");
        if (ctRunning_)
            stopCtAcquisition();
        ctHardwareStopPending_ = true;
        ctRecoveryPending_ = true;
        ctRecoveryConfirmed_ = false;
        if (kohzuController_)
            QMetaObject::invokeMethod(kohzuController_, "recoverAfterInterruptedSession", Qt::QueuedConnection);
        setMosaicLocked(true);
    }
}
void MainWindow::advanceMosaicPositioning()
{
    if (!mosaicRunning_ || ctRunning_ || !mosaicPendingAxis_.isEmpty())
        return;
    if (mosaicIndex_ >= mosaicPlan_.tiles.size())
    {
        writeMosaicManifest(true);
        if (!mosaicRunning_)
            return;
        writeCtRecoveryState("COMPLETED", "Mosaic acquisition finished.");
        mosaicRunning_ = false;
        setMosaicLocked(false);
        ctSavePathEdit_->setText(mosaicRoot_);
        ctStatusLabel_->setText("Mosaic acquisition complete");
        appendLog("All mosaic tiles acquired. Automatic fusion will start when all tile reconstructions finish.");
        maybeStartMosaicFusion();
        if (mosaicRestoreLive_)
            toggleCameraLivePreview();
        return;
    }
    const QStringList axes{"R", "X0", "X90", "Z", "Y"};
    if (mosaicPositionStage_ < 5)
    {
        QString name = axes[mosaicPositionStage_];
        double target = mosaicPositionStage_ == 0
                            ? mosaicSnapshot_.ctAcquisitionUi.beginningAngleDeg
                            : (mosaicPositionStage_ == 4
                                   ? mosaicOriginY_
                                   : mosaicOrigin_[mosaicPositionStage_ - 1] +
                                         mosaicPlan_.tiles[mosaicIndex_].motorOffset[mosaicPositionStage_ - 1]);
        auto &a = motorAxes_[name];
        ++mosaicPositionStage_;
        mosaicPendingAxis_ = name;
        mosaicPendingLabel_ = QString("Mosaic tile %1 %2").arg(mosaicIndex_ + 1).arg(name);
        mosaicMoveAccepted_ = false;
        mosaicMoveActiveSeen_ = false;
        mosaicVerifyingPosition_ = false;
        mosaicPendingTargetPulse_ = qint64(std::llround(target * a.pulsePerUnit));
        if (std::abs(target - a.currentValue) < 0.5 / a.pulsePerUnit)
        {
            mosaicVerifyingPosition_ = true;
            QMetaObject::invokeMethod(kohzuController_, "verifyAxisPosition", Qt::QueuedConnection, Q_ARG(int, a.axis),
                                      Q_ARG(qint64, mosaicPendingTargetPulse_), Q_ARG(qint64, 1),
                                      Q_ARG(QString, mosaicPendingLabel_ + " verify"));
            return;
        }
        const int basePps =
            int(std::clamp<qint64>(std::llround(a.baseSpeedSpin->value() * a.pulsePerUnit), 1, 2500000));
        const int topPps = int(std::max<qint64>(
            2 * basePps, std::clamp<qint64>(std::llround(a.speedSpin->value() * a.pulsePerUnit), 2, 5000000)));
        int acc = int(std::clamp<qint64>(std::llround(a.accelSpin->value() * 100), 1, 10000));
        QMetaObject::invokeMethod(kohzuController_, "configureSpeedTableAndMoveAbsolute", Qt::QueuedConnection,
                                  Q_ARG(int, a.axis), Q_ARG(qint64, qint64(std::llround(target * a.pulsePerUnit))),
                                  Q_ARG(int, 8), Q_ARG(int, basePps), Q_ARG(int, topPps), Q_ARG(int, acc),
                                  Q_ARG(int, acc), Q_ARG(int, 2), Q_ARG(QString, mosaicPendingLabel_ + " speed"),
                                  Q_ARG(QString, mosaicPendingLabel_));
        int gen = mosaicGeneration_;
        QString label = mosaicPendingLabel_;
        QTimer::singleShot(180000, this, [this, gen, label]() {
            if (mosaicRunning_ && gen == mosaicGeneration_ && label == mosaicPendingLabel_)
                abortMosaic("Mosaic positioning timed out.");
        });
        return;
    }
    QJsonObject actual, target;
    for (int j = 0; j < 3; ++j)
    {
        QString axis = axes[j + 1];
        actual[axis] = motorAxes_[axis].currentValue;
        target[axis] = mosaicOrigin_[j] + mosaicPlan_.tiles[mosaicIndex_].motorOffset[j];
        if (std::abs(actual[axis].toDouble() - target[axis].toDouble()) >
            std::max(1. / motorAxes_[axis].pulsePerUnit, 0.1))
        {
            abortMosaic("Mosaic position readback differs from target.");
            return;
        }
    }
    mosaicTileRecords_[mosaicIndex_]["ACTUAL_CONTROLLER_UM"] = actual;
    mosaicTileRecords_[mosaicIndex_]["TARGET_UM"] = target;
    mosaicTileRecords_[mosaicIndex_]["R_DEG"] = motorAxes_["R"].currentValue;
    mosaicTileRecords_[mosaicIndex_]["Y_UM"] = motorAxes_["Y"].currentValue;
    mosaicTileRecords_[mosaicIndex_]["POSITION_SOURCE"] = "Kohzu pulse counter; not an independent encoder";
    double dx = actual["X0"].toDouble() - mosaicOrigin_[0], dy = actual["X90"].toDouble() - mosaicOrigin_[1],
           dz = actual["Z"].toDouble() - mosaicOrigin_[2];
    const auto &calibration = mosaicPlan_.config;
    double angle = calibration.motorToReconDeg * std::acos(-1.) / 180, cs = std::cos(angle), sn = std::sin(angle);
    mosaicTileRecords_[mosaicIndex_]["VOXEL_ORIGIN"] =
        QJsonArray{-(cs * dx - sn * calibration.x90Sign * dy) / calibration.pixelUm,
                   -(sn * dx + cs * calibration.x90Sign * dy) / calibration.pixelUm,
                   -calibration.zSign * dz / calibration.pixelUm};
    writeMosaicManifest(false);
    if (!mosaicRunning_)
        return;
    ctSavePathEdit_->setText(QDir(mosaicRoot_).filePath(mosaicPlan_.tiles[mosaicIndex_].directory));
    ctStepFlatGapSpin_->setValue(0);
    ctStepFlatMoveSpin_->setValue(mosaicSnapshot_.ctAcquisitionUi.stepFlatMoveUm);
    ctStepFlatNumSpin_->setValue(mosaicSnapshot_.ctAcquisitionUi.stepFlatNum);
    mosaicLaunchingTile_ = true;
    startCtAcquisition();
    mosaicLaunchingTile_ = false;
    if (!ctRunning_)
    {
        abortMosaic("Tile acquisition could not start; inspect preceding error.");
        return;
    }
    setMosaicLocked(true);
}
void MainWindow::finishMosaicTile(bool ok, const QString &message)
{
    if (!mosaicRunning_)
        return;
    if (ok && (!actualCtImageSizeKnown_ || actualCtImageWidth_ != mosaicPlan_.width ||
               actualCtImageHeight_ != mosaicPlan_.height))
    {
        abortMosaic("Camera dimensions differ from mosaic plan; recalibrate the plan for the actual ROI.");
        return;
    }
    mosaicTileRecords_[mosaicIndex_]["COMPLETED"] = ok;
    mosaicTileRecords_[mosaicIndex_]["ERROR"] = ok ? QString() : message;
    writeMosaicManifest(false, ok ? QString() : message);
    if (!mosaicRunning_)
        return;
    if (!ok)
    {
        abortMosaic(message);
        return;
    }
    ++mosaicIndex_;
    mosaicPositionStage_ = 0;
    setMosaicLocked(true);
    int gen = mosaicGeneration_;
    QTimer::singleShot(500, this, [this, gen]() {
        if (mosaicRunning_ && gen == mosaicGeneration_)
            advanceMosaicPositioning();
    });
}
void MainWindow::abortMosaic(const QString &message)
{
    if (!mosaicRunning_)
        return;
    mosaicAutoFusionManifest_.clear();
    writeMosaicManifest(false, message);
    mosaicRunning_ = false;
    ++mosaicGeneration_;
    mosaicPendingAxis_.clear();
    mosaicPendingLabel_.clear();
    ctHardwareStopPending_ = true;
    ctRecoveryPending_ = true;
    ctRecoveryConfirmed_ = false;
    writeCtRecoveryState("STOPPING", message);
    if (ctRunning_)
        stopCtAcquisition();
    if (kohzuController_)
        QMetaObject::invokeMethod(kohzuController_, "recoverAfterInterruptedSession", Qt::QueuedConnection);
    setMosaicLocked(true);
    ctSavePathEdit_->setText(mosaicRoot_);
    appendLog("Mosaic stopped: " + message);
}
void MainWindow::launchMosaicStitch(bool synthetic)
{
    if (mosaicRunning_ || ctRunning_ || controller_.isBusy() ||
        (mosaicProcess_ && mosaicProcess_->state() != QProcess::NotRunning))
    {
        appendLog("Finish current acquisition/reconstruction before mosaic processing.");
        return;
    }
    QString exe = QDir(QCoreApplication::applicationDirPath()).filePath("ct_mosaic.exe");
    if (!QFileInfo::exists(exe))
    {
        appendLog("ct_mosaic.exe is missing.");
        return;
    }
    QStringList args;
    if (synthetic)
    {
        QString root = QFileDialog::getExistingDirectory(this, "Select parent directory for synthetic test");
        if (root.isEmpty())
            return;
        args << "--self-test"
             << QDir(root).filePath("mosaic_test_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss_zzz"));
    }
    else
    {
        QString name = mosaicOutputName_->text().trimmed();
        if (name.isEmpty() || name == "." || name == ".." || name.contains('/') || name.contains('\\'))
        {
            appendLog("Choose a new output folder name inside the mosaic sample.");
            return;
        }
        QString manifest = mosaicManifestEdit_->text();
        args << "--stitch" << manifest << QDir(QFileInfo(manifest).absolutePath()).filePath(name)
             << mosaicTileReconName_->text();
        if (!mosaicRegistration_->text().isEmpty())
            args << mosaicRegistration_->text();
    }
    if (mosaicProcess_)
        mosaicProcess_->deleteLater();
    mosaicProcess_ = new QProcess(this);
    mosaicProcess_->setProcessChannelMode(QProcess::MergedChannels);
    connect(mosaicProcess_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        appendLog("Mosaic process error: " + mosaicProcess_->errorString());
        mosaicTestButton_->setEnabled(true);
        mosaicStitchButton_->setEnabled(true);
    });
    connect(mosaicProcess_, &QProcess::readyReadStandardOutput, this, [this]() {
        auto lines = QString::fromUtf8(mosaicProcess_->readAllStandardOutput()).split('\n', Qt::SkipEmptyParts);
        for (auto &line : lines)
            appendLog(line.trimmed());
    });
    connect(mosaicProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus) {
                appendLog(QString("Mosaic process finished: exit %1.").arg(code));
                mosaicTestButton_->setEnabled(true);
                mosaicStitchButton_->setEnabled(true);
            });
    mosaicTestButton_->setEnabled(false);
    mosaicStitchButton_->setEnabled(false);
    mosaicProcess_->start(exe, args);
}

void MainWindow::queueMosaicReconstruction()
{
    if (mosaicRunning_ || ctRunning_ || controller_.isBusy() || !reconQueuePaths_.isEmpty() ||
        (mosaicProcess_ && mosaicProcess_->state() != QProcess::NotRunning))
    {
        appendLog("Finish active jobs before re-queueing a mosaic.");
        return;
    }
    auto read = [](const QString &path) {
        QFile f(path);
        if (!f.open(QIODevice::ReadOnly))
            throw std::runtime_error(("Cannot read " + path).toStdString());
        const auto doc = QJsonDocument::fromJson(f.readAll());
        if (!doc.isObject())
            throw std::runtime_error("Invalid mosaic JSON.");
        return doc.object();
    };
    try
    {
        const QString manifest = mosaicManifestEdit_->text().trimmed();
        const auto m = read(manifest);
        if (m.value("TYPE").toString() != "CT_STREAM_MOSAIC" || !m.value("COMPLETED").toBool())
            throw std::runtime_error("Select a completed mosaic manifest.");
        const QString root = QFileInfo(manifest).absolutePath();
        const QString folder = mosaicTileReconName_->text().trimmed();
        if (folder.isEmpty() || folder == "." || folder == ".." || folder.contains('/') || folder.contains('\\'))
            throw std::runtime_error("Use a new tile reconstruction folder name, e.g. ReconAlpha20.");
        const QString output = mosaicOutputName_->text().trimmed();
        if (output.isEmpty() || output == "." || output == ".." || output.contains('/') || output.contains('\\') ||
            QFileInfo::exists(QDir(root).filePath(output)))
            throw std::runtime_error("Choose a new mosaic output folder name.");
        const auto config = MosaicConfig::fromJson(m.value("CONFIG").toObject());
        QString frozen = mosaicRegistration_->text().trimmed();
        const QString defaultGeometry = QDir(root).filePath("MosaicRecon/reconstruction.json");
        if (frozen.isEmpty() && QFileInfo::exists(defaultGeometry))
            frozen = defaultGeometry;
        if (!frozen.isEmpty())
        {
            const auto prior = read(frozen);
            const QString fingerprint = QString::fromLatin1(
                QCryptographicHash::hash(QJsonDocument(m).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256)
                    .toHex());
            if (!prior.value("COMPLETED").toBool() || prior.value("SOURCE_SHA256").toString() != fingerprint)
                throw std::runtime_error("Reuse registration must be a completed result for this same mosaic.");
            mosaicRegistration_->setText(frozen);
        }
        const auto tiles = m.value("TILES").toArray();
        if (tiles.isEmpty())
            throw std::runtime_error("Mosaic contains no tiles.");
        QVector<QPair<QString, QJsonObject>> jobs;
        for (auto v : tiles)
        {
            const QString name = v.toObject().value("DIRECTORY").toString();
            if (name.isEmpty() || name.contains('/') || name.contains('\\') || name == "..")
                throw std::runtime_error("Invalid tile directory.");
            const QString projection = QDir(root).filePath(name);
            const auto scan = read(QDir(projection).filePath("scan_manifest.json"));
            if (!scan.value("COMPLETED").toBool())
                throw std::runtime_error("Incomplete tile scan.");
            const QString out = QDir(projection).filePath(folder);
            if (QFileInfo::exists(out))
                throw std::runtime_error("Tile output already exists; choose a new folder name.");
            AppConfig cfg = configFromUi();
            cfg.io.saveDepth = SaveDepth::Bit32;
            cfg.io.saveFormat = "tiff";
            cfg.phase.pixelSizeUm = float(config.pixelUm);
            cfg.geometry.corAuto = true;
            cfg.geometry.corShift = 0;
            if (cfg.io.reconMethod == ReconMethod::RealTimeFBP)
                cfg.io.reconMethod = ReconMethod::OfflineFBP;
            if (m.contains("CALIBRATED_RECON_METHOD") &&
                m.value("CALIBRATED_RECON_METHOD").toString() != toString(cfg.io.reconMethod))
                throw std::runtime_error("Mosaic axis mapping was calibrated for a different reconstruction method.");
            cfg.io.watchDir = projection;
            cfg.io.saveDir = out;
            jobs.append({QDir(out).filePath("reconstruction.json"),
                         QJsonObject{{"TYPE", "CT_STREAM_RECON_JOB"},
                                     {"VERSION", 1},
                                     {"STATUS", "queued"},
                                     {"JOB_SOURCE", "MOSAIC_RECON"},
                                     {"PROJECTION_DIR", projection},
                                     {"OUTPUT_DIR", out},
                                     {"SETTINGS", QJsonObject{{"RECON_CONFIG", cfg.toJsonObject()}}}}});
        }
        for (auto &job : jobs)
        {
            if (!QDir().mkpath(QFileInfo(job.first).absolutePath()) || !atomicJson(job.first, job.second))
                throw std::runtime_error("Could not save a tile reconstruction job.");
        }
        for (auto &job : jobs)
            reconQueuePaths_.append(job.first);
        mosaicAutoFusionManifest_ = manifest;
        mosaicAutoFusionTileRecon_ = folder;
        mosaicAutoFusionOutput_ = output;
        mosaicAutoFusionRegistration_ = frozen;
        refreshReconQueue();
        startNextQueuedReconstruction();
        appendLog(QString("Queued %1 float tile reconstructions; fusion will follow automatically.").arg(jobs.size()));
    }
    catch (const std::exception &e)
    {
        appendLog("Mosaic requeue failed: " + QString::fromUtf8(e.what()));
    }
}
void MainWindow::maybeStartMosaicFusion()
{
    if (mosaicAutoFusionManifest_.isEmpty() || mosaicRunning_ || ctRunning_ || controller_.isBusy() ||
        !reconQueuePaths_.isEmpty() || !activeQueueJobPath_.isEmpty() ||
        (mosaicProcess_ && mosaicProcess_->state() != QProcess::NotRunning))
        return;
    mosaicManifestEdit_->setText(mosaicAutoFusionManifest_);
    mosaicTileReconName_->setText(mosaicAutoFusionTileRecon_);
    mosaicOutputName_->setText(mosaicAutoFusionOutput_);
    mosaicRegistration_->setText(mosaicAutoFusionRegistration_);
    mosaicAutoFusionManifest_.clear();
    launchMosaicStitch(false);
}
void MainWindow::runMosaicQueueSmoke()
{
    if (!qEnvironmentVariableIsSet("CT_UI_SMOKE"))
        return;
    const QString root = qEnvironmentVariable("CT_MOSAIC_QUEUE_TEST");
    QFile f(QDir(root).filePath("test_config.json"));
    if (!f.open(QIODevice::ReadOnly))
    {
        QCoreApplication::exit(4);
        return;
    }
    applyConfig(AppConfig::fromJsonObject(QJsonDocument::fromJson(f.readAll()).object()));
    mosaicManifestEdit_->setText(QDir(root).filePath("mosaic_manifest.json"));
    mosaicTileReconName_->setText(qEnvironmentVariable("CT_MOSAIC_TEST_TILE_RECON", "Recon"));
    mosaicOutputName_->setText(qEnvironmentVariable("CT_MOSAIC_TEST_OUTPUT", "MosaicRecon"));
    mosaicRegistration_->setText(qEnvironmentVariable("CT_MOSAIC_TEST_REGISTRATION"));
    queueMosaicReconstruction();
    auto *watch = new QTimer(this);
    auto *deadline = new QTimer(this);
    deadline->setSingleShot(true);
    connect(deadline, &QTimer::timeout, this, []() { QCoreApplication::exit(6); });
    deadline->start(120000);
    connect(watch, &QTimer::timeout, this, [this, root, watch, deadline]() {
        if (!mosaicProcess_ || mosaicProcess_->state() != QProcess::NotRunning)
            return;
        watch->stop();
        deadline->stop();
        QFile report(QDir(root).filePath(mosaicOutputName_->text() + "/reconstruction.json"));
        bool ok = mosaicProcess_->exitCode() == 0 && report.open(QIODevice::ReadOnly) &&
                  QJsonDocument::fromJson(report.readAll()).object().value("COMPLETED").toBool();
        if (ok)
            close();
        else
            QCoreApplication::exit(5);
    });
    watch->start(100);
}
} // namespace ct
