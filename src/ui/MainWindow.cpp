#include "ui/MainWindow.h"

#include "core/common/PreviewSliceRenderer.h"
#include "core/io/ImageFileIo.h"
#include "ui/ContrastHistogramWidget.h"
#include "ui/ZoomPanImageView.h"

#include <QAbstractSpinBox>
#include <QCloseEvent>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDoubleSpinBox>
#include <QDir>
#include <QEvent>
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
#include <QDateTime>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QSpinBox>
#include <QStringList>
#include <QSplitter>
#include <QTextEdit>
#include <QTabWidget>
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

constexpr int kMotorSpeedColumnsLeftOffset = 70 + 78 + 34 + 8;
constexpr int kMotorSpeedColumnWidth = 92;
constexpr int kMotorSpeedColumnSpacing = 4;
constexpr int kMotorManualSpeedTable = 8;
constexpr int kCtStepSpeedTable = kMotorManualSpeedTable;
// Kohzu table slots are mutable profiles, not permanently assigned "step" or
// "fly" modes.  A fly acquisition therefore uses the same slot in two
// phases: the normal positioning profile is loaded for the initial/lead
// moves, then the slot is rewritten with the constant-speed profile immediately
// before the formal sweep.  Keeping these names separate makes that hand-off
// explicit and prevents the slow scan profile from being used for a several-
// hundred-degree lead move.
constexpr int kCtFlySpeedTable = kMotorManualSpeedTable;
constexpr int kKohzuSpeedPattern = 2;
constexpr qint64 kReconQueueCameraSettleMs = 1000;
constexpr int kPreviewClipCheckMaxSamples = 4096;
constexpr double kPreviewClipResetFraction = 0.995;
constexpr double kCtFlyLeadMarginDeg = 3.0;
constexpr int kCtTabStep = 0;
constexpr int kCtTabTiming = 1;
constexpr int kCtTabFly = 2;
constexpr int kCtTabTimeSeries = 3;
// The pool contains only frames waiting for/being written by the TIFF
// pipeline; it is not a projection archive.  A capacity of 128 gives the
// camera and writer enough burst headroom without reserving tens of GiB for a
// full-frame (3200x3200x16-bit) acquisition.  The writer queue is derived
// from this same bound, so a slow disk applies back-pressure instead of
// turning every projection into a permanent RAM allocation.
constexpr int kCtFramePoolCapacity = 128;

bool previewWindowProducesClippedImage(const cv::Mat& image, const PreviewWindow& window)
{
    if (image.empty() || !window.isValid())
    {
        return false;
    }

    const int total = image.rows * image.cols;
    if (total <= 0)
    {
        return false;
    }

    const int stride = std::max(1, total / kPreviewClipCheckMaxSamples);
    int finiteCount = 0;
    int lowClipped = 0;
    int highClipped = 0;
    int index = 0;
    for (int y = 0; y < image.rows; ++y)
    {
        const float* row = image.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x, ++index)
        {
            if ((index % stride) != 0)
            {
                continue;
            }
            const float value = row[x];
            if (!std::isfinite(value))
            {
                continue;
            }
            ++finiteCount;
            if (value <= window.lower)
            {
                ++lowClipped;
            }
            else if (value >= window.upper)
            {
                ++highClipped;
            }
        }
    }

    if (finiteCount < 64)
    {
        return false;
    }

    const double lowFraction = static_cast<double>(lowClipped) / static_cast<double>(finiteCount);
    const double highFraction = static_cast<double>(highClipped) / static_cast<double>(finiteCount);
    return lowFraction >= kPreviewClipResetFraction || highFraction >= kPreviewClipResetFraction;
}

void writeStartupTrace(const QString& text)
{
    QFile file(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("ct_startup_trace.log")));
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        QTextStream stream(&file);
        stream << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << " " << text << '\n';
    }
}

class IgnoreWheelFilter : public QObject
{
public:
    explicit IgnoreWheelFilter(QObject* parent)
        : QObject(parent)
    {
    }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        if (event->type() == QEvent::Wheel)
        {
            event->ignore();
            return true;
        }
        return QObject::eventFilter(watched, event);
    }
};

template <typename T>
T* makeSpin(QWidget* parent, double min, double max, double step)
{
    auto* spin = new T(parent);
    spin->setRange(min, max);
    spin->setSingleStep(step);
    // Numeric fields are edited directly in the UI.  Hide the native
    // up/down button column consistently for both QSpinBox and
    // QDoubleSpinBox (including Padding, Snap, CT mode and reconstruction
    // controls); keyboard entry and the configured range remain unchanged.
    spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    spin->installEventFilter(new IgnoreWheelFilter(spin));
    return spin;
}

bool isKohzuErrorLog(const QString& text)
{
    const QString lower = text.toLower();
    // Keep only actionable controller failures.  Routine connection,
    // position/status polling, command traces, successful WTB/APS and normal
    // idle/recovery messages are intentionally omitted from the run log.
    static const QStringList errorMarkers = {
        QStringLiteral(" failed"),
        QStringLiteral("failure"),
        QStringLiteral(" error"),
        QStringLiteral(" timeout"),
        QStringLiteral("timed out"),
        QStringLiteral("not connected"),
        QStringLiteral("not idle"),
        QStringLiteral("not confirmed"),
        QStringLiteral("incomplete"),
        QStringLiteral("blocked"),
        QStringLiteral("retry"),
        QStringLiteral("busy"),
        QStringLiteral("605"),
        QStringLiteral("803"),
        QStringLiteral("unavailable"),
        QStringLiteral("unrecognized"),
        QStringLiteral("not understood")
    };
    for (const QString& marker : errorMarkers)
    {
        if (lower.contains(marker))
        {
            return true;
        }
    }
    return false;
}

bool isRoutineMotorLog(const QString& text)
{
    const QString lower = text.toLower();
    // These are high-frequency, successful motion/status messages.  They do
    // not describe a fault and are intentionally kept out of the UI/run log;
    // command failures still pass through isKohzuErrorLog above.
    static const QStringList motorIndicators = {
        QStringLiteral("kohzu"),
        QStringLiteral("motor"),
        QStringLiteral("axis "),
        QStringLiteral("axes "),
        QStringLiteral(" rdp"),
        QStringLiteral(" str"),
        QStringLiteral(" aps"),
        QStringLiteral(" wtb"),
        QStringLiteral(" stp"),
        QStringLiteral("speed table"),
        QStringLiteral("position polling")
    };
    bool motorMessage = false;
    for (const QString& marker : motorIndicators)
    {
        if (lower.contains(marker))
        {
            motorMessage = true;
            break;
        }
    }
    if (motorMessage && !isKohzuErrorLog(text))
    {
        return true;
    }

    static const QStringList routineMessages = {
        QStringLiteral("kohzu tcp connected"),
        QStringLiteral("kohzu idn:"),
        QStringLiteral("kohzu connected:"),
        QStringLiteral("kohzu controller verified"),
        QStringLiteral("kohzu initial position refresh"),
        QStringLiteral("kohzu command #"),
        QStringLiteral("kohzu rdp polling trace"),
        QStringLiteral("axis "),
        QStringLiteral("ct motor setup:"),
        QStringLiteral("ct fly scan motor sweep started:"),
        QStringLiteral("ct fly r scan move completed."),
        QStringLiteral("ct fly device-frame gate armed"),
        QStringLiteral("stopping all kohzu axes"),
        QStringLiteral("kohzu idle all-axis polling paused."),
        QStringLiteral("kohzu idle all-axis polling resumed."),
        QStringLiteral("kohzu recovery: issuing controlled stop"),
        QStringLiteral("kohzu recovery completed"),
        QStringLiteral("ct recovery: controller connected; checking r/y idle state."),
        QStringLiteral("ct recovery confirmed; new acquisitions are enabled.")
    };
    for (const QString& marker : routineMessages)
    {
        if (lower.contains(marker) && !isKohzuErrorLog(text))
        {
            return true;
        }
    }
    return false;
}

QComboBox* makeCombo(QWidget* parent)
{
    auto* combo = new QComboBox(parent);
    combo->installEventFilter(new IgnoreWheelFilter(combo));
    return combo;
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
    double defaultBaseSpeed;
    double defaultSpeed;
    double defaultAccel;
};

const MotorAxisDefinition kMotorAxes[] = {
    { "X90", "parallel sample trim", "XA04A-R102", 16, "CH8-2", "um", 2.0, 10.0, 500.0, -5000.0, 5000.0, 400.0, 2000.0, 0.5 },
    { "X0", "vertical sample trim", "XA04A-R102", 15, "CH8-1", "um", 2.0, 10.0, 500.0, -5000.0, 5000.0, 400.0, 2000.0, 0.5 },
    { "R", "CT rotation", "RA10A-W01", 17, "CH9-1", "deg", 20000.0, 0.01, 1.0, -800.0, 800.0, 2.0, 15.0, 1.5 },
    { "U", "goniometer vertical beam", "SA10A-RM01", 1, "CH1-1", "deg", 1190.4761904762, 0.001, 0.1, -5.0, 4.0, 0.8, 0.8, 0.5 },
    { "V", "goniometer parallel beam", "SA10A-RB01", 2, "CH1-2", "deg", 1492.5373134328, 0.001, 0.1, -5.0, 4.0, 0.8, 0.8, 0.5 },
    { "Z", "sample height", "ZA16A-32F01", 18, "CH9-2", "um", 10.0, 1.0, 100.0, -25000.0, 25000.0, 160.0, 800.0, 0.5 },
    { "Y", "large horizontal stage", "XA16F-L2101", 3, "CH2-1", "um", 0.2, 10.0, 1000.0, -100000.0, 50000.0, 200.0, 2000.0, 0.5 },
    { "Det_OT2_Z", "Kinetix detector vertical lift", "ZA16A-32F01", 22, "CH11-2", "um", 10.0, 1.0, 100.0, -25000.0, 25000.0, 100.0, 500.0, 0.5 },
    { "Det_OT2_Y", "Kinetix detector horizontal travel", "XA16F-L2301", 9, "CH5-1", "um", 0.2, 10.0, 1000.0, -152000.0, 152000.0, 100.0, 500.0, 0.5 },
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

QString formatCtSeconds(qint64 ms)
{
    return QString::number(static_cast<double>(std::max<qint64>(0, ms)) / 1000.0, 'f', 3);
}

QString formatDurationSeconds(double seconds)
{
    const double safeSeconds = std::max(0.0, seconds);
    if (safeSeconds >= 60.0)
    {
        return QStringLiteral("%1 s (%2 min)")
            .arg(safeSeconds, 0, 'f', 3)
            .arg(safeSeconds / 60.0, 0, 'f', 2);
    }
    return QStringLiteral("%1 s").arg(safeSeconds, 0, 'f', 3);
}

QString formatClockDuration(double seconds)
{
    const qint64 totalSeconds = static_cast<qint64>(std::llround(std::max(0.0, seconds)));
    const qint64 hours = totalSeconds / 3600;
    const qint64 minutes = (totalSeconds % 3600) / 60;
    const qint64 secs = totalSeconds % 60;
    if (hours > 0)
    {
        return QStringLiteral("%1:%2:%3")
            .arg(hours)
            .arg(minutes, 2, 10, QLatin1Char('0'))
            .arg(secs, 2, 10, QLatin1Char('0'));
    }
    return QStringLiteral("%1:%2")
        .arg(minutes)
        .arg(secs, 2, 10, QLatin1Char('0'));
}

void addTimingMs(QJsonObject& obj, const QString& key, qint64 ms)
{
    const qint64 clamped = std::max<qint64>(0, ms);
    obj[key + QStringLiteral("_MS")] = static_cast<double>(clamped);
    obj[key + QStringLiteral("_S")] = static_cast<double>(clamped) / 1000.0;
}

QLabel* makeCenteredHeader(const QString& text, QWidget* parent)
{
    auto* label = new QLabel(text, parent);
    label->setAlignment(Qt::AlignCenter);
    return label;
}

struct MotorSpeedTableParams
{
    int speedTable = 0;
    int startSpeedPps = 1;
    int topSpeedPps = 2;
    int accelTime10ms = 1;
    int decelTime10ms = 1;
    int accelPattern = kKohzuSpeedPattern;
    QString summary;
};

MotorSpeedTableParams makeMotorSpeedTableParams(const QString& axisName, int speedTable, double pulsePerUnit,
    const QString& unit, double startSpeedPerSec, double topSpeedPerSec, double accelTimeSec)
{
    MotorSpeedTableParams params;
    params.speedTable = speedTable;
    const double safePulsePerUnit = std::max(1e-9, pulsePerUnit);
    const double safeStart = std::max(0.0, startSpeedPerSec);
    const double safeTop = std::max(0.0, topSpeedPerSec);

    qint64 startPps = std::llround(safeStart * safePulsePerUnit);
    qint64 topPps = std::llround(safeTop * safePulsePerUnit);
    startPps = std::clamp<qint64>(startPps, 1, 2500000);
    topPps = std::clamp<qint64>(topPps, 2, 5000000);
    // ARIES speed tables require the configured top speed to leave enough
    // headroom above the start speed for the acceleration profile.  In the
    // field the controller rejects a table with top < 2*start as WTB 605
    // even though the two values are otherwise positive and ordered (for
    // example 11110/22219).  The fly-plan calculation is floating point, so
    // rounding can land exactly one pulse below that boundary.  Normalize
    // the pair before sending WTB; this changes the commanded speed by at
    // most one pulse in the usual case and avoids a controller-side reject.
    const qint64 minimumTopPps = std::min<qint64>(5000000, startPps * 2);
    if (topPps < minimumTopPps)
    {
        topPps = minimumTopPps;
    }
    if (topPps <= startPps)
    {
        if (startPps < 5000000)
        {
            topPps = startPps + 1;
        }
        else
        {
            startPps = topPps - 1;
        }
    }

    const double safeAccelTimeSec = std::max(0.01, accelTimeSec);
    const int accelTicks = static_cast<int>(std::clamp<qint64>(std::llround(safeAccelTimeSec * 100.0), 1, 10000));

    params.startSpeedPps = static_cast<int>(startPps);
    params.topSpeedPps = static_cast<int>(topPps);
    params.accelTime10ms = accelTicks;
    params.decelTime10ms = accelTicks;
    params.summary = QStringLiteral("%1 speed ok: %2 -> %3 %4/s, acc %5 s")
        .arg(axisName)
        .arg(safeStart, 0, 'f', 4)
        .arg(safeTop, 0, 'f', 4)
        .arg(unit)
        .arg(static_cast<double>(params.accelTime10ms) / 100.0, 0, 'f', 2);
    return params;
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

double exposureMsFromSettings(const PvcamCaptureSettings& settings)
{
    switch (settings.exposureUnit)
    {
    case PvcamExposureUnit::Microseconds:
        return settings.exposureValue / 1000.0;
    case PvcamExposureUnit::Seconds:
        return settings.exposureValue * 1000.0;
    case PvcamExposureUnit::Milliseconds:
    default:
        return settings.exposureValue;
    }
}

QString cameraSnapPathForIndex(const QString& outputDir, const QString& filePrefix, int filePadding, int index)
{
    const QString prefix = filePrefix.isEmpty() ? QStringLiteral("kinetix_frame_") : filePrefix;
    const int padding = std::clamp(filePadding, 1, 12);
    QDir dir(outputDir);
    return dir.filePath(QStringLiteral("%1%2.tiff")
        .arg(prefix)
        .arg(std::max(0, index), padding, 10, QLatin1Char('0')));
}

QString numberedPath(const QString& dirPath, const QString& prefix, int padding, int index, const QString& ext)
{
    QDir dir(dirPath);
    return dir.filePath(QStringLiteral("%1%2.%3")
        .arg(prefix)
        .arg(std::max(0, index), std::clamp(padding, 1, 12), 10, QLatin1Char('0'))
        .arg(ext));
}

cv::Mat materializeCameraFrameImage(const CameraFrame& frame)
{
    if (!frame.image.empty())
    {
        return frame.image;
    }
    if (!frame.bufferLease || frame.bufferLease->data() == nullptr
        || frame.width <= 0 || frame.height <= 0 || frame.strideBytes <= 0)
    {
        return {};
    }
    return cv::Mat(frame.height, frame.width, frame.cvType,
        frame.bufferLease->data(), static_cast<size_t>(frame.strideBytes));
}

bool directoryHasAnyEntries(const QString& path)
{
    QDir dir(path);
    if (!dir.exists())
    {
        return false;
    }
    return !dir.entryList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System).isEmpty();
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
    obj["RAW_FRAME_LIMIT"] = settings.rawFrameLimit;
    obj["FRAME_INTERVAL_MS"] = settings.frameIntervalMs;
    obj["FIRST_KEEP_DELAY_MS"] = settings.firstKeepDelayMs;
    obj["ALLOW_FRAME_INTERVAL_DECIMATION"] = settings.allowFrameIntervalDecimation;
    obj["RAW_FRAME_STRIDE"] = settings.rawFrameStride;
    obj["BASELINE_RAW_FRAME_PERIOD_MS"] = settings.baselineRawFramePeriodMs;
    obj["FRAME_POOL_CAPACITY"] = settings.framePoolCapacity;
    obj["RING_FRAME_COUNT"] = settings.ringFrameCount;
    obj["WAIT_FOR_FRAME_GATE"] = settings.waitForFrameGate;
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

    QJsonObject phase;
    phase["ENABLE"] = cfg.phase.enable;
    phase["ENERGY_KEV"] = cfg.phase.energyKev;
    phase["DISTANCE_M"] = cfg.phase.distanceM;
    phase["PIXEL_SIZE_UM"] = cfg.phase.pixelSizeUm;
    phase["ALPHA_RATIO"] = cfg.phase.alphaRatio;
    job["PHASE"] = phase;

    QJsonObject stripe;
    stripe["ENABLE"] = false;
    stripe["METHOD"] = cfg.stripe.method.isEmpty() ? QStringLiteral("FW") : cfg.stripe.method;
    stripe["LEVEL"] = cfg.stripe.level;
    stripe["WNAME"] = cfg.stripe.waveletName.isEmpty() ? QStringLiteral("db5") : cfg.stripe.waveletName;
    stripe["SIGMA"] = cfg.stripe.sigma;
    stripe["PAD"] = cfg.stripe.pad;
    stripe["NCHUNK"] = cfg.stripe.chunk;
    job["STRIPE"] = stripe;

    QJsonObject preprocess;
    preprocess["TAPER_WIDTH"] = cfg.preprocess.taperWidth;
    preprocess["FILTER_TYPE"] = toString(cfg.preprocess.filterType);
    preprocess["FLAT_GAP"] = cfg.preprocess.flatGap;
    preprocess["FLAT_NUM"] = cfg.preprocess.flatNum;
    if (!cfg.preprocess.flatGroups.isEmpty())
    {
        QJsonArray flatGroups;
        for (const FlatGroupConfig& group : cfg.preprocess.flatGroups)
        {
            QJsonObject obj;
            obj["LABEL"] = group.label;
            obj["AFTER_PROJECTION_COUNT"] = group.afterProjectionCount;
            obj["FIRST_FLAT_INDEX"] = group.firstFlatIndex;
            obj["FLAT_COUNT"] = group.flatCount;
            flatGroups.append(obj);
        }
        preprocess["FLAT_GROUPS"] = flatGroups;
    }
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

QJsonArray flatGroupConfigsToJson(const QVector<FlatGroupConfig>& groups)
{
    QJsonArray array;
    for (const FlatGroupConfig& group : groups)
    {
        QJsonObject obj;
        obj["LABEL"] = group.label;
        obj["AFTER_PROJECTION_COUNT"] = group.afterProjectionCount;
        obj["FIRST_FLAT_INDEX"] = group.firstFlatIndex;
        obj["FLAT_COUNT"] = group.flatCount;
        array.append(obj);
    }
    return array;
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
    const QJsonObject phase = job.value(QStringLiteral("PHASE")).toObject();
    if (!phase.isEmpty())
    {
        cfg.phase.enable = phase.value(QStringLiteral("ENABLE")).toBool(cfg.phase.enable);
        cfg.phase.energyKev = static_cast<float>(phase.value(QStringLiteral("ENERGY_KEV")).toDouble(cfg.phase.energyKev));
        cfg.phase.distanceM = static_cast<float>(phase.value(QStringLiteral("DISTANCE_M")).toDouble(cfg.phase.distanceM));
        cfg.phase.pixelSizeUm = static_cast<float>(phase.value(QStringLiteral("PIXEL_SIZE_UM")).toDouble(cfg.phase.pixelSizeUm));
        cfg.phase.alphaRatio = static_cast<float>(phase.value(QStringLiteral("ALPHA_RATIO")).toDouble(cfg.phase.alphaRatio));
    }
    const QJsonObject stripe = job.value(QStringLiteral("STRIPE")).toObject();
    if (!stripe.isEmpty())
    {
        cfg.stripe.enable = stripe.value(QStringLiteral("ENABLE")).toBool(cfg.stripe.enable);
        cfg.stripe.method = stripe.value(QStringLiteral("METHOD")).toString(cfg.stripe.method).toUpper();
        if (cfg.stripe.method.isEmpty())
        {
            cfg.stripe.method = QStringLiteral("FW");
        }
        cfg.stripe.level = std::clamp(stripe.value(QStringLiteral("LEVEL")).toInt(cfg.stripe.level), 0, 20);
        cfg.stripe.waveletName = stripe.value(QStringLiteral("WNAME")).toString(cfg.stripe.waveletName);
        if (cfg.stripe.waveletName.isEmpty())
        {
            cfg.stripe.waveletName = QStringLiteral("db5");
        }
        cfg.stripe.sigma = std::max(0.01f, static_cast<float>(stripe.value(QStringLiteral("SIGMA")).toDouble(cfg.stripe.sigma)));
        cfg.stripe.pad = stripe.value(QStringLiteral("PAD")).toBool(cfg.stripe.pad);
        cfg.stripe.chunk = std::clamp(stripe.value(QStringLiteral("NCHUNK")).toInt(cfg.stripe.chunk), 1, 1024);
        cfg.stripe.enable = false;
    }
    const QJsonObject preprocess = job.value(QStringLiteral("PREPROCESS")).toObject();
    if (!preprocess.isEmpty())
    {
        cfg.preprocess.method = PreprocessMethod::Log;
        cfg.preprocess.airValue = 1.0f;
        cfg.preprocess.taperWidth = preprocess.value(QStringLiteral("TAPER_WIDTH")).toInt(cfg.preprocess.taperWidth);
        cfg.preprocess.filterType = filterTypeFromString(preprocess.value(QStringLiteral("FILTER_TYPE")).toString(toString(cfg.preprocess.filterType)));
        cfg.preprocess.flatGap = std::max(0, preprocess.value(QStringLiteral("FLAT_GAP")).toInt(cfg.preprocess.flatGap));
        cfg.preprocess.flatNum = std::max(1, preprocess.value(QStringLiteral("FLAT_NUM")).toInt(cfg.preprocess.flatNum));
        const QJsonArray flatGroups = preprocess.value(QStringLiteral("FLAT_GROUPS")).toArray();
        if (!flatGroups.isEmpty())
        {
            cfg.preprocess.flatGroups.clear();
            for (const QJsonValue& value : flatGroups)
            {
                const QJsonObject obj = value.toObject();
                FlatGroupConfig group;
                group.label = obj.value(QStringLiteral("LABEL")).toString();
                group.afterProjectionCount = obj.value(QStringLiteral("AFTER_PROJECTION_COUNT")).toInt();
                group.firstFlatIndex = obj.value(QStringLiteral("FIRST_FLAT_INDEX")).toInt();
                group.flatCount = obj.value(QStringLiteral("FLAT_COUNT")).toInt();
                if (group.flatCount > 0)
                {
                    cfg.preprocess.flatGroups.append(group);
                }
            }
        }
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

int extractStartIndexFromPattern(const QString& pattern)
{
    const QRegularExpression numberRegex(QStringLiteral(R"((\d+))"));
    QRegularExpressionMatchIterator it = numberRegex.globalMatch(pattern);
    int value = 0;
    while (it.hasNext())
    {
        value = it.next().captured(1).toInt();
    }
    return value;
}

QVector<FlatGroupConfig> inferQueueFlatGroups(const AppConfig& cfg, int firstFlatIndex)
{
    QVector<FlatGroupConfig> groups;
    const int flatNum = std::max(1, cfg.preprocess.flatNum);

    auto appendGroup = [&](const QString& label, int afterProjectionCount) {
        FlatGroupConfig group;
        group.label = label;
        group.afterProjectionCount = std::clamp(afterProjectionCount, 0, cfg.geometry.totalAngles);
        group.firstFlatIndex = firstFlatIndex + groups.size() * flatNum;
        group.flatCount = flatNum;
        groups.append(group);
    };

    appendGroup(QStringLiteral("start"), 0);
    if (cfg.preprocess.flatGap > 0)
    {
        for (int after = cfg.preprocess.flatGap; after < cfg.geometry.totalAngles; after += cfg.preprocess.flatGap)
        {
            appendGroup(QStringLiteral("interval"), after);
        }
        appendGroup(QStringLiteral("end"), cfg.geometry.totalAngles);
    }
    return groups;
}

int findFirstFlatIndexForConfig(const AppConfig& cfg)
{
    const QDir projectionDir(cfg.io.watchDir);
    for (int index = 0; index < 100000; ++index)
    {
        const QString flatPath = projectionDir.filePath(numberedFileName(cfg.io.flatPrefix,
            cfg.io.inputPadding, index, cfg.io.inputExt));
        if (QFileInfo::exists(flatPath))
        {
            return index;
        }
    }
    return -1;
}

int findFirstTomoIndexForConfig(const AppConfig& cfg)
{
    const QDir projectionDir(cfg.io.watchDir);
    for (int index = 0; index < 100000; ++index)
    {
        const QString tomoPath = projectionDir.filePath(numberedFileName(cfg.io.tomoPrefix,
            cfg.io.inputPadding, index, cfg.io.inputExt));
        if (QFileInfo::exists(tomoPath))
        {
            return index;
        }
    }
    return -1;
}

int resolveTomoStartIndexForConfig(AppConfig& cfg)
{
    const QDir projectionDir(cfg.io.watchDir);
    const auto hasCompleteTomoRange = [&](int startIndex) {
        const QString firstTomo = projectionDir.filePath(numberedFileName(cfg.io.tomoPrefix,
            cfg.io.inputPadding, startIndex, cfg.io.inputExt));
        const QString lastTomo = projectionDir.filePath(numberedFileName(cfg.io.tomoPrefix,
            cfg.io.inputPadding, startIndex + cfg.geometry.totalAngles - 1, cfg.io.inputExt));
        return QFileInfo::exists(firstTomo) && QFileInfo::exists(lastTomo);
    };

    const int configuredStartIndex = extractStartIndexFromPattern(cfg.io.filePattern0);
    if (hasCompleteTomoRange(configuredStartIndex))
    {
        return configuredStartIndex;
    }

    const int detectedStartIndex = findFirstTomoIndexForConfig(cfg);
    if (detectedStartIndex >= 0 && hasCompleteTomoRange(detectedStartIndex))
    {
        cfg.io.filePattern0 = numberedFileName(cfg.io.tomoPrefix,
            cfg.io.inputPadding, detectedStartIndex, cfg.io.inputExt);
        return detectedStartIndex;
    }

    return configuredStartIndex;
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

QString pixelValueText(const cv::Mat& image, int x, int y)
{
    if (image.empty() || x < 0 || y < 0 || x >= image.cols || y >= image.rows)
    {
        return QStringLiteral("-");
    }
    switch (image.depth())
    {
    case CV_8U:
        return QString::number(image.at<uchar>(y, x));
    case CV_16U:
        return QString::number(image.at<ushort>(y, x));
    case CV_16S:
        return QString::number(image.at<short>(y, x));
    case CV_32S:
        return QString::number(image.at<int>(y, x));
    case CV_32F:
        return QString::number(static_cast<double>(image.at<float>(y, x)), 'g', 7);
    case CV_64F:
        return QString::number(image.at<double>(y, x), 'g', 7);
    default:
        return QStringLiteral("type%1").arg(image.type());
    }
}

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    writeStartupTrace(QStringLiteral("MainWindow ctor begin"));
    setWindowTitle(QStringLiteral("C++ CT Workstation v1"));
    resize(1850, 980);
    cameraPreviewFitInitialized_ = false;
    reconPreviewFitInitialized_ = false;
    qRegisterMetaType<ct::CameraFrame>("ct::CameraFrame");
    qRegisterMetaType<ct::CtFrameSaveItem>("ct::CtFrameSaveItem");
    qRegisterMetaType<ct::CtReplayAcquisitionSettings>("ct::CtReplayAcquisitionSettings");
    qRegisterMetaType<ct::PvcamTimingCalibrationSettings>("ct::PvcamTimingCalibrationSettings");
    qRegisterMetaType<ct::PvcamTimingCalibrationResult>("ct::PvcamTimingCalibrationResult");
    qRegisterMetaType<ct::ContinuousTimingStats>("ct::ContinuousTimingStats");
    qRegisterMetaType<ct::ContinuousFrameMetadata>("ct::ContinuousFrameMetadata");
    qRegisterMetaType<QVector<ct::ContinuousFrameMetadata>>("QVector<ct::ContinuousFrameMetadata>");

    auto* central = new QWidget(this);
    auto* mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(8, 8, 8, 8);
    mainLayout->setSpacing(8);
    setCentralWidget(central);

    auto* hardwareScroll = new QScrollArea(central);
    hardwareScroll->setWidgetResizable(true);
    hardwareScroll->setFixedWidth(540);
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
    cameraExposureSpin_ = makeSpin<QDoubleSpinBox>(cameraGroup, 1.0, 600000.0, 1.0);
    cameraExposureSpin_->setValue(10.0);
    cameraExposureSpin_->setDecimals(0);
    cameraExposureSpin_->setButtonSymbols(QAbstractSpinBox::NoButtons);
    cameraExposureUnitCombo_ = makeCombo(cameraGroup);
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
    auto* cameraRoiXLayout = new QHBoxLayout();
    cameraRoiXLayout->setContentsMargins(0, 0, 0, 0);
    cameraRoiXLayout->setSpacing(6);
    cameraRoiXLayout->addWidget(cameraRoiXSpin_, 1);
    cameraRoiXLayout->addWidget(new QLabel(QStringLiteral("Width"), cameraGroup));
    cameraRoiXLayout->addWidget(cameraRoiWidthSpin_, 1);
    auto* cameraRoiXWidget = new QWidget(cameraGroup);
    cameraRoiXWidget->setLayout(cameraRoiXLayout);
    auto* cameraRoiYLayout = new QHBoxLayout();
    cameraRoiYLayout->setContentsMargins(0, 0, 0, 0);
    cameraRoiYLayout->setSpacing(6);
    cameraRoiYLayout->addWidget(cameraRoiYSpin_, 1);
    cameraRoiYLayout->addWidget(new QLabel(QStringLiteral("Height"), cameraGroup));
    cameraRoiYLayout->addWidget(cameraRoiHeightSpin_, 1);
    auto* cameraRoiYWidget = new QWidget(cameraGroup);
    cameraRoiYWidget->setLayout(cameraRoiYLayout);
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
    auto* cameraSnapIndexLayout = new QHBoxLayout();
    cameraSnapIndexLayout->setContentsMargins(0, 0, 0, 0);
    cameraSnapIndexLayout->setSpacing(6);
    cameraSnapIndexLayout->addWidget(cameraSnapPaddingSpin_, 1);
    cameraSnapIndexLayout->addWidget(new QLabel(QStringLiteral("Snap Index"), cameraGroup));
    cameraSnapIndexLayout->addWidget(cameraSnapIndexSpin_, 1);
    auto* cameraSnapIndexWidget = new QWidget(cameraGroup);
    cameraSnapIndexWidget->setLayout(cameraSnapIndexLayout);
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
    cameraForm->addRow(QStringLiteral("ROI X"), cameraRoiXWidget);
    cameraForm->addRow(QStringLiteral("ROI Y"), cameraRoiYWidget);
    cameraForm->addRow(cameraRoiButtonWidget);
    cameraForm->addRow(QStringLiteral("Snap Prefix"), cameraSnapPrefixEdit_);
    cameraForm->addRow(QStringLiteral("Padding"), cameraSnapIndexWidget);
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

    auto* ctGroup = new QGroupBox(QStringLiteral("Acquisition"), hardwareHost);
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
    ctModeTabs_ = new QTabWidget(ctGroup);
    auto* ctStepPage = new QWidget(ctModeTabs_);
    auto* ctStepLayout = new QFormLayout(ctStepPage);
    ctStepLayout->setContentsMargins(6, 6, 6, 6);
    auto* ctStepLabel = new QLabel(QStringLiteral("Step-and-shoot: move R, expose one projection, repeat."), ctStepPage);
    ctStepLabel->setWordWrap(true);
    ctStepFlatGapSpin_ = makeSpin<QSpinBox>(ctStepPage, 0, 20000, 1);
    ctStepFlatGapSpin_->setValue(0);
    ctStepFlatMoveSpin_ = makeSpin<QDoubleSpinBox>(ctStepPage, -500000.0, 500000.0, 100.0);
    ctStepFlatMoveSpin_->setDecimals(4);
    ctStepFlatMoveSpin_->setValue(0.0);
    ctStepFlatNumSpin_ = makeSpin<QSpinBox>(ctStepPage, 0, 10000, 1);
    ctStepFlatNumSpin_->setValue(10);
    ctStepLayout->addRow(ctStepLabel);
    ctStepLayout->addRow(QStringLiteral("Flat Gap"), ctStepFlatGapSpin_);
    ctStepLayout->addRow(QStringLiteral("Flat Move um"), ctStepFlatMoveSpin_);
    ctStepLayout->addRow(QStringLiteral("Flat Num"), ctStepFlatNumSpin_);
    auto* ctTimingPage = new QWidget(ctModeTabs_);
    auto* ctTimingForm = new QFormLayout(ctTimingPage);
    ctTimingForm->setContentsMargins(6, 6, 6, 6);
    ctTimingStatusValueLabel_ = new QLabel(QStringLiteral("Not measured"), ctTimingPage);
    ctTimingSourceValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingExposureValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingRoiValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingReadoutValueLabel_ = new QLabel(QStringLiteral("16-bit Dynamic Range"), ctTimingPage);
    ctTimingSamplesValueLabel_ = new QLabel(QStringLiteral("3 warm-up + 10 measured"), ctTimingPage);
    ctTimingPeriodValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingRateValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingMaxDeviationValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingRangeValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingDriftValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingFitResidualValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingExposureRangeValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingFramesValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingDroppedValueLabel_ = new QLabel(QStringLiteral("-"), ctTimingPage);
    ctTimingCalibrateButton_ = new QPushButton(QStringLiteral("Calibrate Timing"), ctTimingPage);
    ctTimingForm->addRow(QStringLiteral("Timing Status"), ctTimingStatusValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Timing Source"), ctTimingSourceValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Exposure"), ctTimingExposureValueLabel_);
    ctTimingForm->addRow(QStringLiteral("ROI"), ctTimingRoiValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Readout"), ctTimingReadoutValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Samples"), ctTimingSamplesValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Fitted Frame Period"), ctTimingPeriodValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Raw Frame Rate"), ctTimingRateValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Max Period Deviation"), ctTimingMaxDeviationValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Observed Range"), ctTimingRangeValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Mean Drift"), ctTimingDriftValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Max Fit Residual"), ctTimingFitResidualValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Exposure Range"), ctTimingExposureRangeValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Device Frames"), ctTimingFramesValueLabel_);
    ctTimingForm->addRow(QStringLiteral("Dropped Frames"), ctTimingDroppedValueLabel_);
    ctTimingForm->addRow(ctTimingCalibrateButton_);
    auto* ctFlyPage = new QWidget(ctModeTabs_);
    auto* ctFlyForm = new QFormLayout(ctFlyPage);
    ctFlyForm->setContentsMargins(6, 6, 6, 6);
    ctFlyAllowedBlurSpin_ = makeSpin<QDoubleSpinBox>(ctFlyPage, 0.01, 1000.0, 0.1);
    ctFlyAllowedBlurSpin_->setDecimals(3);
    ctFlyAllowedBlurSpin_->setValue(1.0);
    ctFlyFlatMoveSpin_ = makeSpin<QDoubleSpinBox>(ctFlyPage, -500000.0, 500000.0, 100.0);
    ctFlyFlatMoveSpin_->setDecimals(4);
    ctFlyFlatMoveSpin_->setValue(0.0);
    ctFlyFlatNumSpin_ = makeSpin<QSpinBox>(ctFlyPage, 0, 10000, 1);
    ctFlyFlatNumSpin_->setValue(10);
    ctFlyTotalAngleSpin_ = makeSpin<QDoubleSpinBox>(ctFlyPage, 180.0, 180.0, 1.0);
    ctFlyTotalAngleSpin_->setDecimals(4);
    ctFlyTotalAngleSpin_->setValue(180.0);
    ctFlyTotalAngleSpin_->setEnabled(false);
    ctFlyRadiusLabel_ = new QLabel(QStringLiteral("-"), ctFlyPage);
    ctFlySpeedLabel_ = new QLabel(QStringLiteral("-"), ctFlyPage);
    ctFlySpeedLabel_->setToolTip(QStringLiteral("Derived from exposure, ROI radius, and allowed edge blur."));
    ctFlyLeadAngleLabel_ = new QLabel(QStringLiteral("-"), ctFlyPage);
    ctFlyLeadAngleLabel_->setToolTip(QStringLiteral("Automatically derived from R start speed, fly speed, acceleration time, and a fixed 3 deg safety margin."));
    ctFlyScanTimeLabel_ = new QLabel(QStringLiteral("-"), ctFlyPage);
    ctFlyFrameIntervalLabel_ = new QLabel(QStringLiteral("-"), ctFlyPage);
    ctFlyRawStrideLabel_ = new QLabel(QStringLiteral("-"), ctFlyPage);
    ctFlyValidationLabel_ = new QLabel(QStringLiteral("-"), ctFlyPage);
    ctFlyValidationLabel_->setWordWrap(true);
    auto* ctFlyNote = new QLabel(QStringLiteral("Fly v1 uses start/end flats only; interval Flat Gap is ignored."), ctFlyPage);
    ctFlyNote->setWordWrap(true);
    ctFlyForm->addRow(QStringLiteral("ROI Radius px"), ctFlyRadiusLabel_);
    ctFlyForm->addRow(QStringLiteral("Flat Move um"), ctFlyFlatMoveSpin_);
    ctFlyForm->addRow(QStringLiteral("Flat Num"), ctFlyFlatNumSpin_);
    ctFlyForm->addRow(QStringLiteral("Edge Blur px"), ctFlyAllowedBlurSpin_);
    ctFlyForm->addRow(QStringLiteral("Fly Speed deg/s"), ctFlySpeedLabel_);
    ctFlyForm->addRow(QStringLiteral("Scan Time"), ctFlyScanTimeLabel_);
    ctFlyForm->addRow(QStringLiteral("Lead Angle deg"), ctFlyLeadAngleLabel_);
    ctFlyForm->addRow(QStringLiteral("Total Angle deg"), ctFlyTotalAngleSpin_);
    ctFlyForm->addRow(QStringLiteral("Frame Interval"), ctFlyFrameIntervalLabel_);
    ctFlyForm->addRow(QStringLiteral("Raw Frame Stride"), ctFlyRawStrideLabel_);
    ctFlyForm->addRow(QStringLiteral("Fly Check"), ctFlyValidationLabel_);
    ctFlyForm->addRow(ctFlyNote);
    auto* ctTimeSeriesPage = new QWidget(ctModeTabs_);
    auto* ctTimeSeriesForm = new QFormLayout(ctTimeSeriesPage);
    ctTimeSeriesForm->setContentsMargins(6, 6, 6, 6);
    ctTimeSeriesRawFrameStrideSpin_ = makeSpin<QSpinBox>(ctTimeSeriesPage, 1, 1000000, 1);
    ctTimeSeriesRawFrameStrideSpin_->setValue(1);
    ctTimeSeriesActualIntervalLabel_ = new QLabel(QStringLiteral("-"), ctTimeSeriesPage);
    ctTimeSeriesActualRateLabel_ = new QLabel(QStringLiteral("-"), ctTimeSeriesPage);
    ctTimeSeriesDurationLabel_ = new QLabel(QStringLiteral("-"), ctTimeSeriesPage);
    ctTimeSeriesDataSizeLabel_ = new QLabel(QStringLiteral("-"), ctTimeSeriesPage);
    ctTimeSeriesValidationLabel_ = new QLabel(QStringLiteral("Timing calibration required."), ctTimeSeriesPage);
    ctTimeSeriesValidationLabel_->setWordWrap(true);
    auto* ctTimeSeriesNote = new QLabel(QStringLiteral("2D time series: fixed-position continuous imaging; no motor move, flats, COR, or reconstruction job."), ctTimeSeriesPage);
    ctTimeSeriesNote->setWordWrap(true);
    ctTimeSeriesForm->addRow(QStringLiteral("Raw Frame Stride"), ctTimeSeriesRawFrameStrideSpin_);
    ctTimeSeriesForm->addRow(QStringLiteral("Actual Interval"), ctTimeSeriesActualIntervalLabel_);
    ctTimeSeriesForm->addRow(QStringLiteral("Actual Frame Rate"), ctTimeSeriesActualRateLabel_);
    ctTimeSeriesForm->addRow(QStringLiteral("Estimated Duration"), ctTimeSeriesDurationLabel_);
    ctTimeSeriesForm->addRow(QStringLiteral("Estimated Data Size"), ctTimeSeriesDataSizeLabel_);
    ctTimeSeriesForm->addRow(QStringLiteral("2D Check"), ctTimeSeriesValidationLabel_);
    ctTimeSeriesForm->addRow(ctTimeSeriesNote);
    ctModeTabs_->addTab(ctStepPage, QStringLiteral("Step"));
    ctModeTabs_->addTab(ctTimingPage, QStringLiteral("Timing"));
    ctModeTabs_->addTab(ctFlyPage, QStringLiteral("Fly"));
    ctModeTabs_->addTab(ctTimeSeriesPage, QStringLiteral("2D"));
    ctModeTabs_->setTabEnabled(kCtTabFly, false);
    ctModeTabs_->setTabEnabled(kCtTabTimeSeries, false);
    ctTiffThreadsSpin_ = makeSpin<QSpinBox>(ctGroup, 1, kAcquisitionWriterThreadLimit, 1);
    ctTiffThreadsSpin_->setValue(kAcquisitionWriterThreadLimit);
    ctStatusLabel_ = new QLabel(QStringLiteral("Idle"), ctGroup);
    ctTimingLabel_ = new QLabel(QStringLiteral("Elapsed 0:00 / remaining --"), ctGroup);
    ctTimingLabel_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    ctProgressBar_ = new QProgressBar(ctGroup);
    ctProgressBar_->setRange(0, 1000);
    ctProgressBar_->setValue(0);
    ctProgressBar_->setTextVisible(true);
    ctProgressBar_->setFormat(QStringLiteral("%p%"));
    ctProgressBar_->setMaximumHeight(18);
    ctWriterLabel_ = new QLabel(QStringLiteral("16 threads, 0.0 MiB/s, saved 0"), ctGroup);
    ctPvcamRingLabel_ = new QLabel(QStringLiteral("Inactive"), ctGroup);
    ctBufferLabel_ = new QLabel(QStringLiteral("0 / 128 frames, 0.0 / 2.5 GiB"), ctGroup);
    ctFrameTimingLabel_ = new QLabel(QStringLiteral("-"), ctGroup);
    ctFrameTimingLabel_->setWordWrap(true);
    ctStartButton_ = new QPushButton(QStringLiteral("Start"), ctGroup);
    ctStopButton_ = new QPushButton(QStringLiteral("Stop"), ctGroup);
    ctStopButton_->setEnabled(false);
    ctStopButton_->setStyleSheet(QStringLiteral(
        "QPushButton:enabled { background: #b00020; color: white; font-weight: 600; }"
        "QPushButton:disabled { background: #d0d0d0; color: #777777; font-weight: 600; }"));
    auto* ctButtonLayout = new QHBoxLayout();
    ctButtonLayout->setContentsMargins(0, 0, 0, 0);
    ctButtonLayout->addWidget(ctStartButton_);
    ctButtonLayout->addWidget(ctStopButton_);
    auto* ctButtonWidget = new QWidget(ctGroup);
    ctButtonWidget->setLayout(ctButtonLayout);
    auto* ctStatusLayout = new QHBoxLayout();
    ctStatusLayout->setContentsMargins(0, 0, 0, 0);
    ctStatusLayout->setSpacing(8);
    ctStatusLayout->addWidget(ctStatusLabel_, 1);
    ctStatusLayout->addWidget(ctTimingLabel_, 0);
    auto* ctStatusWidget = new QWidget(ctGroup);
    ctStatusWidget->setLayout(ctStatusLayout);
    ctForm->addRow(QStringLiteral("Beginning Angle"), ctBeginningAngleSpin_);
    ctForm->addRow(QStringLiteral("Mode"), ctModeTabs_);
    ctForm->addRow(QStringLiteral("Save Path"), ctSaveWidget);
    ctForm->addRow(QStringLiteral("Projection Num"), ctProjectionNumSpin_);
    ctForm->addRow(QStringLiteral("TIFF Writer Threads"), ctTiffThreadsSpin_);
    ctForm->addRow(QStringLiteral("Status"), ctStatusWidget);
    ctForm->addRow(QStringLiteral("Progress"), ctProgressBar_);
    ctForm->addRow(QStringLiteral("Writer"), ctWriterLabel_);
    ctForm->addRow(QStringLiteral("PVCAM Ring"), ctPvcamRingLabel_);
    ctForm->addRow(QStringLiteral("Buffer"), ctBufferLabel_);
    ctForm->addRow(QStringLiteral("Frame Timing"), ctFrameTimingLabel_);
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
    auto* speedHeaderLayout = new QHBoxLayout(motorSpeedHeader_);
    speedHeaderLayout->setContentsMargins(0, 0, 0, 0);
    speedHeaderLayout->setSpacing(kMotorSpeedColumnSpacing);
    auto* speedHeaderSpacer = new QWidget(motorSpeedHeader_);
    speedHeaderSpacer->setFixedWidth(kMotorSpeedColumnsLeftOffset);
    speedHeaderLayout->addWidget(speedHeaderSpacer);
    auto* baseHeader = makeCenteredHeader(QStringLiteral("Start Speed/s"), motorSpeedHeader_);
    auto* speedHeader = makeCenteredHeader(QStringLiteral("Top Speed/s"), motorSpeedHeader_);
    auto* accelHeader = makeCenteredHeader(QStringLiteral("Accel Time*s"), motorSpeedHeader_);
    baseHeader->setFixedWidth(kMotorSpeedColumnWidth);
    speedHeader->setFixedWidth(kMotorSpeedColumnWidth);
    accelHeader->setFixedWidth(kMotorSpeedColumnWidth);
    speedHeaderLayout->addWidget(baseHeader);
    speedHeaderLayout->addWidget(speedHeader);
    speedHeaderLayout->addWidget(accelHeader);
    speedHeaderLayout->addStretch();
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
        axisUi.baseSpeedSpin = makeSpin<QDoubleSpinBox>(axisGroup, 0.0001, 1000000.0, def.defaultBaseSpeed);
        configureMotorSpinBox(axisUi.baseSpeedSpin, 4, kMotorSpeedColumnWidth);
        axisUi.baseSpeedSpin->setValue(def.defaultBaseSpeed);
        axisUi.baseSpeedSpin->setToolTip(QStringLiteral("Kohzu WTB start speed in %1/s.").arg(axisUi.unit));
        axisUi.speedSpin = makeSpin<QDoubleSpinBox>(axisGroup, 0.0001, 1000000.0, def.defaultSpeed);
        configureMotorSpinBox(axisUi.speedSpin, 4, kMotorSpeedColumnWidth);
        axisUi.speedSpin->setValue(def.defaultSpeed);
        axisUi.speedSpin->setToolTip(QStringLiteral("Kohzu WTB top speed in %1/s.").arg(axisUi.unit));
        axisUi.accelSpin = makeSpin<QDoubleSpinBox>(axisGroup, 0.01, 100.0, 0.01);
        configureMotorSpinBox(axisUi.accelSpin, 2, kMotorSpeedColumnWidth);
        axisUi.accelSpin->setValue(def.defaultAccel);
        axisUi.accelSpin->setToolTip(QStringLiteral("Kohzu WTB acceleration/deceleration time in seconds. Sent as 10 ms ticks."));
        axisUi.speedUnitLabel->hide();
        auto* speedLayout = new QHBoxLayout();
        speedLayout->setContentsMargins(0, 0, 0, 0);
        speedLayout->setSpacing(kMotorSpeedColumnSpacing);
        speedLayout->addWidget(axisUi.baseSpeedSpin);
        speedLayout->addWidget(axisUi.speedSpin);
        speedLayout->addWidget(axisUi.accelSpin);
        auto* speedWidget = new QWidget(axisGroup);
        speedWidget->setLayout(speedLayout);
        speedWidget->setFixedWidth(kMotorSpeedColumnWidth * 3 + kMotorSpeedColumnSpacing * 2);
        speedWidget->hide();
        axisUi.speedWidget = speedWidget;

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
        axisLayout->addWidget(axisUi.speedWidget, axisRow, 3, 1, 3);

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
        if (axisKey == QStringLiteral("R") && axisUi.speedSpin)
        {
            connect(axisUi.speedSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
                updateCtFlyDerivedParams();
            });
        }

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
    reconQueueCurrentButton_ = new QPushButton(QStringLiteral("Queue Current Recon"), reconQueueGroup);
    reconQueueImportButton_ = new QPushButton(QStringLiteral("Import Job"), reconQueueGroup);
    reconQueueButtonLayout->addWidget(reconQueueUpButton_);
    reconQueueButtonLayout->addWidget(reconQueueDownButton_);
    reconQueueButtonLayout->addWidget(reconQueueRemoveButton_);
    reconQueueLayout->addLayout(reconQueueButtonLayout);
    reconQueueLayout->addWidget(reconQueueImportButton_);
    reconQueueLayout->addWidget(reconQueueCurrentButton_);
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
    correctionModeCombo_ = makeCombo(ioGroup);
    correctionModeCombo_->addItems({ "None", "Flat", "Flat-Dark" });
    ioForm->addRow(QStringLiteral("Correction"), correctionModeCombo_);
    reconMethodCombo_ = makeCombo(ioGroup);
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
    extCombo_ = makeCombo(ioGroup);
    extCombo_->addItems({ "tif", "tiff" });
    ioForm->addRow(QStringLiteral("Padding"), paddingSpin_);
    ioForm->addRow(QStringLiteral("Ext"), extCombo_);
    file0Edit_ = new QLineEdit(ioGroup);
    file180Edit_ = new QLineEdit(ioGroup);
    ioForm->addRow(QStringLiteral("0 Deg"), file0Edit_);
    ioForm->addRow(QStringLiteral("180 Deg"), file180Edit_);
    savePrefixEdit_ = new QLineEdit(ioGroup);
    saveDepthCombo_ = makeCombo(ioGroup);
    saveDepthCombo_->addItems({ "8bit", "16bit", "32bit" });
    saveFormatCombo_ = makeCombo(ioGroup);
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
    stripeEnableCheck_ = new QCheckBox(QStringLiteral("Enable FW Stripe"), preprocessGroup);
    stripeLevelSpin_ = makeSpin<QSpinBox>(preprocessGroup, 0, 20, 1);
    stripeSigmaSpin_ = makeSpin<QDoubleSpinBox>(preprocessGroup, 0.01, 20.0, 0.1);
    stripeSigmaSpin_->setDecimals(3);
    stripeEnableCheck_->setToolTip(QStringLiteral("FW stripe/ring artifact suppression is temporarily disabled for beamline commissioning."));
    stripeLevelSpin_->setToolTip(QStringLiteral("FW stripe/ring artifact suppression is temporarily disabled for beamline commissioning."));
    stripeSigmaSpin_->setToolTip(QStringLiteral("FW stripe/ring artifact suppression is temporarily disabled for beamline commissioning."));
    stripeEnableCheck_->hide();
    stripeLevelSpin_->hide();
    stripeSigmaSpin_->hide();
    taperSpin_ = makeSpin<QSpinBox>(preprocessGroup, 0, 200, 1);
    flatGapSpin_ = makeSpin<QSpinBox>(preprocessGroup, 0, 20000, 1);
    flatNumSpin_ = makeSpin<QSpinBox>(preprocessGroup, 1, 10000, 1);
    filterCombo_ = makeCombo(preprocessGroup);
    filterCombo_->addItems({ "shepp-logan", "ram-lak", "cosine", "hamming", "hann" });
    preprocessForm->addRow(stripeEnableCheck_);
    preprocessForm->addRow(QStringLiteral("Stripe Level"), stripeLevelSpin_);
    preprocessForm->addRow(QStringLiteral("Stripe Sigma"), stripeSigmaSpin_);
    preprocessForm->setRowVisible(stripeEnableCheck_, false);
    preprocessForm->setRowVisible(stripeLevelSpin_, false);
    preprocessForm->setRowVisible(stripeSigmaSpin_, false);
    preprocessForm->addRow(QStringLiteral("Taper"), taperSpin_);
    preprocessForm->addRow(QStringLiteral("Flat Gap"), flatGapSpin_);
    preprocessForm->addRow(QStringLiteral("Flat Num"), flatNumSpin_);
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
    cameraPixelLabel_ = new QLabel(QStringLiteral("Camera x=- y=- value=-"), cameraPanel);
    cameraPixelLabel_->setStyleSheet("color:#dddddd;background:#2a2a2a;padding:3px 6px;border-radius:3px;");
    cameraPanelLayout->addWidget(cameraPixelLabel_);
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
    reconPixelLabel_ = new QLabel(QStringLiteral("Recon x=- y=- value=-"), reconPanel);
    reconPixelLabel_->setStyleSheet("color:#dddddd;background:#2a2a2a;padding:3px 6px;border-radius:3px;");
    reconPanelLayout->addWidget(reconPixelLabel_);

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

    writeStartupTrace(QStringLiteral("MainWindow UI constructed; starting camera worker"));
    startCameraWorker();
    writeStartupTrace(QStringLiteral("Camera worker started; starting motor worker"));
    startMotorWorker();
    writeStartupTrace(QStringLiteral("Motor worker started; scheduling motor connect"));
    updateMotorUiEnabled();
    QTimer::singleShot(0, this, &MainWindow::requestMotorConnect);

    writeStartupTrace(QStringLiteral("Connecting UI signals"));
    connect(cameraSnapButton_, &QPushButton::clicked, this, &MainWindow::snapCameraFrame);
    connect(cameraLiveButton_, &QPushButton::clicked, this, &MainWindow::toggleCameraLivePreview);
    connect(cameraApplyRoiButton_, &QPushButton::clicked, this, &MainWindow::applyCameraRoi);
    connect(cameraResetRoiButton_, &QPushButton::clicked, this, &MainWindow::resetCameraRoiFullFrame);
    connect(cameraPreviewView_, &ZoomPanImageView::imagePositionHovered, this, &MainWindow::updateCameraPixelHover);
    connect(cameraPreviewView_, &ZoomPanImageView::imagePositionLeft, this, [this]() {
        if (cameraPixelLabel_) { cameraPixelLabel_->setText(QStringLiteral("Camera x=- y=- value=-")); }
    });
    connect(reconPreviewView_, &ZoomPanImageView::imagePositionHovered, this, &MainWindow::updateReconPixelHover);
    connect(reconPreviewView_, &ZoomPanImageView::imagePositionLeft, this, [this]() {
        if (reconPixelLabel_) { reconPixelLabel_->setText(QStringLiteral("Recon x=- y=- value=-")); }
    });
    connect(ctStartButton_, &QPushButton::clicked, this, &MainWindow::startCtAcquisition);
    connect(ctStopButton_, &QPushButton::clicked, this, &MainWindow::stopCtAcquisition);
    connect(ctTimingCalibrateButton_, &QPushButton::clicked, this, &MainWindow::startTimingCalibration);
    connect(ctModeTabs_, &QTabWidget::currentChanged, this, [this](int) {
        if (!ctRunning_)
        {
            ctContinuousTimingStats_ = {};
            ctContinuousTimingDriftLevel_ = 0;
        }
        updateCtModeUiState();
        updateCtFlyDerivedParams();
        updateCtTimeSeriesDerivedParams();
        updateContinuousTimingUi();
    });
    connect(cameraExposureSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
        invalidateTimingCalibration(QStringLiteral("Exposure changed."));
        updateCtFlyDerivedParams();
        updateCtTimeSeriesDerivedParams();
    });
    connect(cameraExposureUnitCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) {
        invalidateTimingCalibration(QStringLiteral("Exposure unit changed."));
        updateCtFlyDerivedParams();
        updateCtTimeSeriesDerivedParams();
    });
    connect(cameraRoiXSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
        invalidateTimingCalibration(QStringLiteral("ROI changed."));
    });
    connect(cameraRoiYSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
        invalidateTimingCalibration(QStringLiteral("ROI changed."));
    });
    connect(cameraRoiWidthSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
        invalidateTimingCalibration(QStringLiteral("ROI changed."));
        updateCtFlyDerivedParams();
        updateCtTimeSeriesDerivedParams();
    });
    connect(cameraRoiHeightSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
        invalidateTimingCalibration(QStringLiteral("ROI changed."));
        updateCtFlyDerivedParams();
        updateCtTimeSeriesDerivedParams();
    });
    connect(ctProjectionNumSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
        updateCtFlyDerivedParams();
        updateCtTimeSeriesDerivedParams();
    });
    connect(ctFlyAllowedBlurSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
        updateCtFlyDerivedParams();
    });
    if (ctTimeSeriesRawFrameStrideSpin_)
    {
        connect(ctTimeSeriesRawFrameStrideSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
            updateCtTimeSeriesDerivedParams();
            updateCtProgressUi();
        });
    }
    connect(reconQueueUpButton_, &QPushButton::clicked, this, [this]() { moveSelectedReconQueueJob(-1); });
    connect(reconQueueDownButton_, &QPushButton::clicked, this, [this]() { moveSelectedReconQueueJob(1); });
    connect(reconQueueRemoveButton_, &QPushButton::clicked, this, &MainWindow::removeSelectedReconQueueJob);
    connect(reconQueueImportButton_, &QPushButton::clicked, this, &MainWindow::importReconJobs);
    connect(reconQueueCurrentButton_, &QPushButton::clicked, this, &MainWindow::enqueueCurrentReconJob);
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

    ctProgressTimer_ = new QTimer(this);
    ctProgressTimer_->setInterval(1000);
    connect(ctProgressTimer_, &QTimer::timeout, this, &MainWindow::updateCtProgressUi);
    updateCtProgressUi();

    const QString sessionConfigFile = sessionConfigPath();
    writeStartupTrace(QStringLiteral("Preparing recon queue"));
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
    writeStartupTrace(QStringLiteral("Loading session config"));
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
    QString timingCacheError;
    if (!timingProfileCache_.load(timingProfileCachePath(), &timingCacheError))
    {
        appendLog(QStringLiteral("Timing profile cache unavailable; starting empty: %1").arg(timingCacheError));
    }
    else if (restoreTimingProfileForCurrentUi(true))
    {
        appendLog(QStringLiteral("Timing profile cache loaded: %1 profile(s).").arg(timingProfileCache_.size()));
    }
    appendLog(QStringLiteral("App executable: %1").arg(QCoreApplication::applicationFilePath()));
    appendLog(QStringLiteral("Build stamp: %1 %2")
        .arg(QString::fromLatin1(__DATE__), QString::fromLatin1(__TIME__)));
    inspectInterruptedCtSession();
    writeStartupTrace(QStringLiteral("MainWindow ctor complete"));
    QTimer::singleShot(0, this, &MainWindow::startNextQueuedReconstruction);
    QTimer::singleShot(5000, this, [this]() {
        if (cameraWarmOpenComplete_)
        {
            return;
        }
        cameraWarmOpenComplete_ = true;
        cameraWarmOpenCompletedMs_ = QDateTime::currentMSecsSinceEpoch();
        writeStartupTrace(QStringLiteral("camera warm open fallback timeout; allowing recon queue"));
        if (reconStartDeferredForCameraWarmOpen_)
        {
            reconStartDeferredForCameraWarmOpen_ = false;
            QTimer::singleShot(static_cast<int>(kReconQueueCameraSettleMs), this, &MainWindow::startNextQueuedReconstruction);
        }
    });
    QTimer::singleShot(1000, this, &MainWindow::maybeStartCtSmokeAutostart);
    QTimer::singleShot(1000, this, &MainWindow::maybeStartCtReplayAutostart);
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
    applyingConfig_ = true;
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
    stripeEnableCheck_->setChecked(config.stripe.enable);
    stripeLevelSpin_->setValue(config.stripe.level);
    stripeSigmaSpin_->setValue(config.stripe.sigma);
    taperSpin_->setValue(config.preprocess.taperWidth);
    flatGapSpin_->setValue(config.preprocess.flatGap);
    flatNumSpin_->setValue(config.preprocess.flatNum);
    filterCombo_->setCurrentText(toString(config.preprocess.filterType));
    batchSpin_->setValue(config.compute.batchSize);
    zChunkSpin_->setValue(config.compute.zChunkSize);
    gpuSpin_->setValue(config.compute.gpuIndex);
    zBlockSpin_->setValue(config.compute.zBlockSplit);
    livePreviewCheck_->setChecked(config.preview.enableLive);
    cameraExposureSpin_->setValue(config.cameraUi.exposureValue);
    cameraExposureUnitCombo_->setCurrentText(config.cameraUi.exposureUnit);
    cameraRoiXSpin_->setValue(config.cameraUi.roiX);
    cameraRoiYSpin_->setValue(config.cameraUi.roiY);
    cameraRoiWidthSpin_->setValue(config.cameraUi.roiWidth);
    cameraRoiHeightSpin_->setValue(config.cameraUi.roiHeight);
    cameraSnapPrefixEdit_->setText(config.cameraUi.snapPrefix);
    cameraSnapPaddingSpin_->setValue(config.cameraUi.snapPadding);
    cameraSnapIndexSpin_->setValue(config.cameraUi.snapIndex);
    cameraSaveDirEdit_->setText(config.cameraUi.saveDir);
    motorHostEdit_->setText(config.kohzuUi.host);
    motorPortSpin_->setValue(config.kohzuUi.port);
    ctBeginningAngleSpin_->setValue(config.ctAcquisitionUi.beginningAngleDeg);
    if (ctModeTabs_)
    {
        const int tabIndex = config.ctAcquisitionUi.modeIndex == 1
            ? kCtTabFly
            : (config.ctAcquisitionUi.modeIndex == 2 ? kCtTabTimeSeries : kCtTabStep);
        ctModeTabs_->setCurrentIndex(tabIndex);
    }
    ctSavePathEdit_->setText(config.ctAcquisitionUi.savePath);
    ctProjectionNumSpin_->setValue(config.ctAcquisitionUi.projectionNum);
    ctStepFlatGapSpin_->setValue(config.ctAcquisitionUi.stepFlatGap);
    ctStepFlatMoveSpin_->setValue(config.ctAcquisitionUi.stepFlatMoveUm);
    ctStepFlatNumSpin_->setValue(config.ctAcquisitionUi.stepFlatNum);
    ctFlyFlatMoveSpin_->setValue(config.ctAcquisitionUi.flyFlatMoveUm);
    ctFlyFlatNumSpin_->setValue(config.ctAcquisitionUi.flyFlatNum);
    if (ctTimeSeriesRawFrameStrideSpin_)
    {
        ctTimeSeriesRawFrameStrideSpin_->setValue(config.ctAcquisitionUi.rawFrameStride);
    }
    ctTiffThreadsSpin_->setValue(config.ctAcquisitionUi.tiffThreads);
    if (ctFlyAllowedBlurSpin_)
    {
        ctFlyAllowedBlurSpin_->setValue(config.ctFly.allowedBlurPx);
    }
    setPreviewSliceRange(config.geometry.volH);
    syncPreviewSliceControls(config.preview.liveSliceIndex);
    for (auto it = motorAxes_.begin(); it != motorAxes_.end(); ++it)
    {
        const auto motorIt = config.motorAxes.constFind(it.key());
        if (motorIt == config.motorAxes.constEnd())
        {
            continue;
        }
        if (it->relativeSpin && motorIt.value().hasRelativeStep)
        {
            it->relativeSpin->setValue(std::abs(motorIt.value().relativeStep));
        }
        if (it->absoluteSpin && motorIt.value().hasAbsoluteTarget)
        {
            it->absoluteSpin->setValue(motorIt.value().absoluteTarget);
        }
        if (it->baseSpeedSpin && motorIt.value().hasBaseSpeed)
        {
            it->baseSpeedSpin->setValue(motorIt.value().baseSpeed);
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
    updateCtModeUiState();
    updateReconMethodUiState();
    updateCtFlyDerivedParams();
    applyingConfig_ = false;
    if (!restoreTimingProfileForCurrentUi(false))
    {
        timingCalibrationValid_ = false;
        timingCalibrationFromCache_ = false;
    }
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
    cfg.phase.energyKev = static_cast<float>(energySpin_->value());
    cfg.phase.distanceM = static_cast<float>(distanceSpin_->value());
    cfg.phase.pixelSizeUm = static_cast<float>(pixelSpin_->value());
    cfg.phase.alphaRatio = static_cast<float>(alphaSpin_->value());
    cfg.stripe.enable = false;
    cfg.stripe.method = QStringLiteral("FW");
    cfg.stripe.level = stripeLevelSpin_->value();
    cfg.stripe.waveletName = QStringLiteral("db5");
    cfg.stripe.sigma = static_cast<float>(stripeSigmaSpin_->value());
    cfg.stripe.pad = true;
    cfg.stripe.chunk = 16;
    cfg.preprocess.method = PreprocessMethod::Log;
    cfg.preprocess.airValue = 1.0f;
    cfg.preprocess.taperWidth = taperSpin_->value();
    cfg.preprocess.flatGap = flatGapSpin_->value();
    cfg.preprocess.flatNum = flatNumSpin_->value();
    cfg.preprocess.filterType = filterTypeFromString(filterCombo_->currentText());
    cfg.compute.batchSize = batchSpin_->value();
    cfg.compute.zChunkSize = zChunkSpin_->value();
    cfg.compute.gpuIndex = gpuSpin_->value();
    cfg.compute.zBlockSplit = zBlockSpin_->value();
    cfg.preview.enableLive = livePreviewCheck_->isChecked();
    cfg.preview.liveSliceIndex = liveSliceSpin_->value();
    cfg.cameraUi.exposureValue = cameraExposureSpin_->value();
    cfg.cameraUi.exposureUnit = cameraExposureUnitCombo_->currentText();
    cfg.cameraUi.roiX = cameraRoiXSpin_->value();
    cfg.cameraUi.roiY = cameraRoiYSpin_->value();
    cfg.cameraUi.roiWidth = cameraRoiWidthSpin_->value();
    cfg.cameraUi.roiHeight = cameraRoiHeightSpin_->value();
    cfg.cameraUi.snapPrefix = cameraSnapPrefixEdit_->text();
    cfg.cameraUi.snapPadding = cameraSnapPaddingSpin_->value();
    cfg.cameraUi.snapIndex = cameraSnapIndexSpin_->value();
    cfg.cameraUi.saveDir = cameraSaveDirEdit_->text();
    cfg.ctAcquisitionUi.beginningAngleDeg = ctBeginningAngleSpin_->value();
    const int selectedTab = ctModeTabs_ ? ctModeTabs_->currentIndex() : kCtTabStep;
    cfg.ctAcquisitionUi.modeIndex = selectedTab == kCtTabFly
        ? 1
        : (selectedTab == kCtTabTimeSeries ? 2 : 0);
    cfg.ctAcquisitionUi.savePath = ctSavePathEdit_->text();
    cfg.ctAcquisitionUi.projectionNum = ctProjectionNumSpin_->value();
    cfg.ctAcquisitionUi.stepFlatGap = ctStepFlatGapSpin_->value();
    cfg.ctAcquisitionUi.stepFlatMoveUm = ctStepFlatMoveSpin_->value();
    cfg.ctAcquisitionUi.stepFlatNum = ctStepFlatNumSpin_->value();
    cfg.ctAcquisitionUi.flyFlatMoveUm = ctFlyFlatMoveSpin_->value();
    cfg.ctAcquisitionUi.flyFlatNum = ctFlyFlatNumSpin_->value();
    cfg.ctAcquisitionUi.flyLeadAngleDeg = ctFlyDerivedLeadAngleDeg_;
    cfg.ctAcquisitionUi.rawFrameStride = ctTimeSeriesRawFrameStrideSpin_
        ? ctTimeSeriesRawFrameStrideSpin_->value()
        : cfg.ctAcquisitionUi.rawFrameStride;
    cfg.ctAcquisitionUi.tiffThreads = ctTiffThreadsSpin_->value();
    cfg.kohzuUi.host = motorHostEdit_->text().trimmed();
    cfg.kohzuUi.port = motorPortSpin_->value();
    cfg.ctFly.allowedBlurPx = ctFlyAllowedBlurSpin_ ? ctFlyAllowedBlurSpin_->value() : cfg.ctFly.allowedBlurPx;
    cfg.motorAxes.clear();
    for (auto it = motorAxes_.cbegin(); it != motorAxes_.cend(); ++it)
    {
        MotorAxisConfig motorCfg;
        motorCfg.hasRelativeStep = true;
        motorCfg.relativeStep = it->relativeSpin ? it->relativeSpin->value() : 0.0;
        motorCfg.hasAbsoluteTarget = true;
        motorCfg.absoluteTarget = it->absoluteSpin ? it->absoluteSpin->value() : 0.0;
        motorCfg.hasBaseSpeed = true;
        motorCfg.baseSpeed = it->baseSpeedSpin ? it->baseSpeedSpin->value() : 0.0;
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

QString MainWindow::timingProfileCachePath() const
{
    return QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("timing_profiles.json"));
}

QString MainWindow::ctRecoveryStatePath() const
{
    return QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("ct_recovery_state.json"));
}

void MainWindow::writeCtRecoveryState(const QString& state, const QString& detail)
{
    QJsonObject stateObject;
    stateObject[QStringLiteral("STATE")] = state;
    stateObject[QStringLiteral("DETAIL")] = detail;
    stateObject[QStringLiteral("UPDATED_AT")] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    if (!writeJsonFileAtomic(ctRecoveryStatePath(), stateObject))
    {
        appendLog(QStringLiteral("CT recovery state write failed: %1").arg(ctRecoveryStatePath()));
    }
}

void MainWindow::inspectInterruptedCtSession()
{
    QJsonObject state;
    QString error;
    if (!readJsonFile(ctRecoveryStatePath(), &state, &error))
    {
        if (QFileInfo::exists(ctRecoveryStatePath()))
        {
            appendLog(QStringLiteral("CT recovery state unreadable: %1").arg(error));
        }
        return;
    }
    const QString value = state.value(QStringLiteral("STATE")).toString();
    if (value == QStringLiteral("RUNNING") || value == QStringLiteral("STOPPING"))
    {
        ctRecoveryPending_ = true;
        appendLog(QStringLiteral("Previous CT session was interrupted at %1. Hardware recovery is required before a new CT.")
            .arg(state.value(QStringLiteral("UPDATED_AT")).toString()));
    }
}

TimingProfileKey MainWindow::timingProfileKeyFromUi() const
{
    TimingProfileKey key;
    key.cameraIdentity = QStringLiteral("Kinetix/PVCAM");
    key.readoutMode = QStringLiteral("16-bit Dynamic Range");
    key.triggerMode = QStringLiteral("internal_timed");
    const PvcamTimingCalibrationSettings settings = timingCalibrationSettingsFromUi();
    PvcamCaptureSettings exposureSettings;
    exposureSettings.exposureValue = settings.exposureValue;
    exposureSettings.exposureUnit = settings.exposureUnit;
    key.exposureMs = exposureMsFromSettings(exposureSettings);
    key.roiX = cameraRoiXSpin_ ? cameraRoiXSpin_->value() : 0;
    key.roiY = cameraRoiYSpin_ ? cameraRoiYSpin_->value() : 0;
    key.roiWidth = cameraRoiWidthSpin_ ? cameraRoiWidthSpin_->value() : 3200;
    key.roiHeight = cameraRoiHeightSpin_ ? cameraRoiHeightSpin_->value() : 3200;
    return key;
}

bool MainWindow::restoreTimingProfileForCurrentUi(bool logHit)
{
    const TimingProfileKey key = timingProfileKeyFromUi();
    const TimingProfile* profile = timingProfileCache_.find(key);
    if (profile == nullptr)
    {
        return false;
    }
    timingCalibrationSettings_ = timingCalibrationSettingsFromUi();
    timingCalibrationResult_ = {};
    timingCalibrationResult_.ok = profile->analysis.valid;
    timingCalibrationResult_.message = QStringLiteral("Timing restored from cached profile.");
    timingCalibrationResult_.timingSource = profile->timingSource;
    timingCalibrationResult_.exposureMs = key.exposureMs;
    timingCalibrationResult_.roiX = key.roiX;
    timingCalibrationResult_.roiY = key.roiY;
    timingCalibrationResult_.roiWidth = key.roiWidth;
    timingCalibrationResult_.roiHeight = key.roiHeight;
    timingCalibrationResult_.bitDepth = profile->bitDepth;
    timingCalibrationResult_.metadataVersion = profile->metadataVersion;
    timingCalibrationResult_.cameraName = profile->cameraName;
    timingCalibrationResult_.cameraPartNumber = profile->cameraPartNumber;
    timingCalibrationResult_.cameraSerial = profile->cameraSerial;
    timingCalibrationResult_.readoutMode = profile->key.readoutMode;
    timingCalibrationResult_.readoutPort = profile->readoutPort;
    timingCalibrationResult_.readoutSpeed = profile->readoutSpeed;
    timingCalibrationResult_.gain = profile->gain;
    timingCalibrationResult_.triggerMode = profile->key.triggerMode;
    timingCalibrationResult_.binningX = profile->key.binningX;
    timingCalibrationResult_.binningY = profile->key.binningY;
    timingCalibrationResult_.analysis = profile->analysis;
    timingCalibrationFromCache_ = true;
    if (logHit)
    {
        appendLog(QStringLiteral("Timing cache hit: %1 fps, period=%2 ms, ROI=%3x%4, exposure=%5 ms.")
            .arg(profile->analysis.rawFrameRateFps, 0, 'f', 3)
            .arg(profile->analysis.fittedPeriodMs, 0, 'f', 6)
            .arg(key.roiWidth)
            .arg(key.roiHeight)
            .arg(key.exposureMs, 0, 'f', 3));
    }
    handleTimingCalibrationResult(timingCalibrationResult_);
    if (timingProfileCache_.markUsed(key, QDateTime::currentDateTime().toString(Qt::ISODateWithMs)))
    {
        QString error;
        if (!timingProfileCache_.saveAtomic(timingProfileCachePath(), &error))
        {
            appendLog(QStringLiteral("Failed to update timing_profiles.json last-used time: %1").arg(error));
        }
    }
    return timingCalibrationValid_;
}

void MainWindow::storeTimingProfile(const PvcamTimingCalibrationResult& result)
{
    if (!result.ok || !result.analysis.valid)
    {
        return;
    }
    TimingProfile profile;
    profile.key = timingProfileKeyFromUi();
    profile.cameraName = result.cameraName;
    profile.cameraPartNumber = result.cameraPartNumber;
    profile.cameraSerial = result.cameraSerial;
    profile.bitDepth = result.bitDepth;
    profile.metadataVersion = result.metadataVersion;
    profile.timingSource = result.timingSource;
    profile.readoutPort = result.readoutPort;
    profile.readoutSpeed = result.readoutSpeed;
    profile.gain = result.gain;
    profile.analysis = result.analysis;
    profile.calibratedAt = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    profile.lastUsedAt = profile.calibratedAt;
    timingProfileCache_.upsert(std::move(profile));
    QString error;
    if (!timingProfileCache_.saveAtomic(timingProfileCachePath(), &error))
    {
        appendLog(QStringLiteral("Failed to save timing_profiles.json: %1").arg(error));
    }
}

void MainWindow::setBusy(bool busy)
{
    writeStartupTrace(QStringLiteral("setBusy enter: %1").arg(busy));
    if (startButton_)
    {
        startButton_->setEnabled(!busy);
    }
    if (stopButton_)
    {
        stopButton_->setEnabled(busy);
        stopButton_->setText(QStringLiteral("Stop"));
    }
    if (saveButton_)
    {
        saveButton_->setEnabled(!busy && controller_.hasVolume());
    }
    writeStartupTrace(QStringLiteral("setBusy exit: %1").arg(busy));
}

PvcamTimingCalibrationSettings MainWindow::timingCalibrationSettingsFromUi() const
{
    PvcamTimingCalibrationSettings settings;
    settings.exposureValue = cameraExposureSpin_ ? cameraExposureSpin_->value() : 10.0;
    settings.exposureUnit = cameraExposureUnitCombo_
        ? cameraExposureUnitFromText(cameraExposureUnitCombo_->currentText())
        : PvcamExposureUnit::Milliseconds;
    settings.roiX = cameraRoiXSpin_ ? cameraRoiXSpin_->value() : 0;
    settings.roiY = cameraRoiYSpin_ ? cameraRoiYSpin_->value() : 0;
    settings.roiWidth = cameraRoiWidthSpin_ ? cameraRoiWidthSpin_->value() : 3200;
    settings.roiHeight = cameraRoiHeightSpin_ ? cameraRoiHeightSpin_->value() : 3200;
    settings.warmupFrames = 3;
    settings.sampleFrames = 10;
    settings.ringFrameCount = 64;
    return settings;
}

bool MainWindow::timingCalibrationMatchesCurrentUi() const
{
    if (!timingCalibrationValid_)
    {
        return false;
    }
    const PvcamTimingCalibrationSettings current = timingCalibrationSettingsFromUi();
    PvcamCaptureSettings currentExposure;
    currentExposure.exposureValue = current.exposureValue;
    currentExposure.exposureUnit = current.exposureUnit;
    PvcamCaptureSettings calibratedExposure;
    calibratedExposure.exposureValue = timingCalibrationSettings_.exposureValue;
    calibratedExposure.exposureUnit = timingCalibrationSettings_.exposureUnit;
    return std::abs(exposureMsFromSettings(currentExposure) - exposureMsFromSettings(calibratedExposure)) < 1e-9
        && current.roiX == timingCalibrationSettings_.roiX
        && current.roiY == timingCalibrationSettings_.roiY
        && current.roiWidth == timingCalibrationSettings_.roiWidth
        && current.roiHeight == timingCalibrationSettings_.roiHeight;
}

void MainWindow::invalidateTimingCalibration(const QString& reason)
{
    const bool wasValid = timingCalibrationValid_;
    timingCalibrationValid_ = false;
    timingCalibrationFromCache_ = false;
    timingInvalidReason_ = reason;
    if (ctTimingStatusValueLabel_)
    {
        ctTimingStatusValueLabel_->setText(timingCalibrationResult_.analysis.valid
            ? QStringLiteral("Stale")
            : QStringLiteral("Not measured"));
        ctTimingStatusValueLabel_->setStyleSheet(QStringLiteral("color:#9a6700;"));
    }
    if (ctTimingExposureValueLabel_)
    {
        PvcamCaptureSettings current;
        current.exposureValue = cameraExposureSpin_ ? cameraExposureSpin_->value() : 0.0;
        current.exposureUnit = cameraExposureUnitCombo_
            ? cameraExposureUnitFromText(cameraExposureUnitCombo_->currentText())
            : PvcamExposureUnit::Milliseconds;
        ctTimingExposureValueLabel_->setText(QStringLiteral("%1 ms").arg(exposureMsFromSettings(current), 0, 'f', 4));
    }
    if (ctTimingRoiValueLabel_)
    {
        ctTimingRoiValueLabel_->setText(QStringLiteral("%1 x %2 @ (%3, %4)")
            .arg(cameraRoiWidthSpin_ ? cameraRoiWidthSpin_->value() : 0)
            .arg(cameraRoiHeightSpin_ ? cameraRoiHeightSpin_->value() : 0)
            .arg(cameraRoiXSpin_ ? cameraRoiXSpin_->value() : 0)
            .arg(cameraRoiYSpin_ ? cameraRoiYSpin_->value() : 0));
    }
    if (!applyingConfig_ && !timingCalibrationRunning_ && !ctRunning_
        && restoreTimingProfileForCurrentUi(true))
    {
        return;
    }
    if (wasValid)
    {
        appendLog(QStringLiteral("Camera timing invalidated: %1 Fly and 2D are disabled.").arg(reason));
    }
    if (ctModeTabs_ && !ctRunning_)
    {
        const int currentTab = ctModeTabs_->currentIndex();
        if (currentTab == kCtTabFly || currentTab == kCtTabTimeSeries)
        {
            ctModeTabs_->setCurrentIndex(kCtTabTiming);
        }
    }
    updateCtModeUiState();
    updateCtFlyDerivedParams();
    updateCtTimeSeriesDerivedParams();
}

void MainWindow::startTimingCalibration()
{
    if (timingCalibrationRunning_ || ctRunning_ || !cameraWorker_)
    {
        return;
    }
    if (cameraSnapSaveInProgress_)
    {
        appendLog(QStringLiteral("Timing calibration cannot start while Snap is running."));
        return;
    }
    if (cameraLiveRunning_)
    {
        timingCalibrationPendingAfterLiveStop_ = true;
        timingRestoreLiveAfterFinish_ = true;
        if (ctTimingStatusValueLabel_)
        {
            ctTimingStatusValueLabel_->setText(QStringLiteral("Stopping Live"));
        }
        stopCameraLivePreview(false);
        return;
    }

    timingCalibrationSettings_ = timingCalibrationSettingsFromUi();
    timingCalibrationRunning_ = true;
    timingCalibrationValid_ = false;
    timingCalibrationFromCache_ = false;
    timingInvalidReason_.clear();
    if (ctTimingStatusValueLabel_)
    {
        ctTimingStatusValueLabel_->setText(QStringLiteral("Measuring"));
        ctTimingStatusValueLabel_->setStyleSheet(QStringLiteral("color:#005fb8;"));
    }
    if (ctTimingSourceValueLabel_) { ctTimingSourceValueLabel_->setText(QStringLiteral("Probing camera")); }
    if (ctTimingCalibrateButton_) { ctTimingCalibrateButton_->setEnabled(false); }
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
    updateCtModeUiState();
    appendLog(QStringLiteral("Starting Kinetix timing calibration."));
    QMetaObject::invokeMethod(cameraWorker_, "calibrateContinuousTiming", Qt::QueuedConnection,
        Q_ARG(ct::PvcamTimingCalibrationSettings, timingCalibrationSettings_));
}

void MainWindow::handleTimingCalibrationResult(const PvcamTimingCalibrationResult& result)
{
    if (shuttingDown_)
    {
        return;
    }
    timingCalibrationRunning_ = false;
    timingCalibrationResult_ = result;
    timingCalibrationValid_ = result.ok;
    if (timingCalibrationValid_ && !timingCalibrationMatchesCurrentUi())
    {
        timingCalibrationValid_ = false;
        timingInvalidReason_ = QStringLiteral("Camera controls changed during calibration.");
    }

    if (ctTimingStatusValueLabel_)
    {
        ctTimingStatusValueLabel_->setText(timingCalibrationValid_
            ? (timingCalibrationFromCache_ ? QStringLiteral("Cached")
                : (result.analysis.stable ? QStringLiteral("Ready") : QStringLiteral("Ready (warning)")))
            : QStringLiteral("Failed"));
        ctTimingStatusValueLabel_->setStyleSheet(!timingCalibrationValid_
            ? QStringLiteral("color:#b00020;")
            : (timingCalibrationFromCache_ || result.analysis.stable
                ? QStringLiteral("color:#1b7f3a;") : QStringLiteral("color:#9a6700;")));
    }
    if (ctTimingSourceValueLabel_)
    {
        ctTimingSourceValueLabel_->setText(result.timingSource == QStringLiteral("embedded_hardware_metadata")
            ? QStringLiteral("Embedded hardware metadata v%1").arg(result.metadataVersion)
            : QStringLiteral("Qualified driver timing"));
    }
    if (ctTimingExposureValueLabel_)
    {
        ctTimingExposureValueLabel_->setText(QStringLiteral("%1 ms").arg(result.exposureMs, 0, 'f', 4));
    }
    if (ctTimingRoiValueLabel_)
    {
        ctTimingRoiValueLabel_->setText(QStringLiteral("%1 x %2 @ (%3, %4)")
            .arg(result.roiWidth).arg(result.roiHeight).arg(result.roiX).arg(result.roiY));
    }
    if (ctTimingReadoutValueLabel_)
    {
        ctTimingReadoutValueLabel_->setText(QStringLiteral("Dynamic Range, %1-bit").arg(result.bitDepth));
    }
    if (ctTimingSamplesValueLabel_)
    {
        ctTimingSamplesValueLabel_->setText(QStringLiteral("%1 warm-up + %2 measured")
            .arg(timingCalibrationSettings_.warmupFrames)
            .arg(result.analysis.sampleCount));
    }
    if (ctTimingPeriodValueLabel_)
    {
        ctTimingPeriodValueLabel_->setText(result.analysis.valid
            ? QStringLiteral("%1 ms").arg(result.analysis.fittedPeriodMs, 0, 'f', 6)
            : QStringLiteral("-"));
    }
    if (ctTimingRateValueLabel_)
    {
        ctTimingRateValueLabel_->setText(result.analysis.valid
            ? QStringLiteral("%1 fps").arg(result.analysis.rawFrameRateFps, 0, 'f', 3)
            : QStringLiteral("-"));
    }
    if (ctTimingMaxDeviationValueLabel_)
    {
        ctTimingMaxDeviationValueLabel_->setText(result.analysis.valid
            ? QStringLiteral("%1 ms").arg(result.analysis.maxPeriodDeviationMs, 0, 'f', 6)
            : QStringLiteral("-"));
    }
    if (ctTimingDriftValueLabel_)
    {
        ctTimingDriftValueLabel_->setText(result.analysis.valid
            ? QStringLiteral("%1 ms (%2%)")
                .arg(result.analysis.meanDriftMs, 0, 'f', 6)
                .arg(result.analysis.meanDriftPercent, 0, 'f', 4)
            : QStringLiteral("-"));
    }
    if (ctTimingFitResidualValueLabel_)
    {
        ctTimingFitResidualValueLabel_->setText(result.analysis.valid
            ? QStringLiteral("%1 ms").arg(result.analysis.maxFitResidualMs, 0, 'f', 6)
            : QStringLiteral("-"));
    }
    if (ctTimingExposureRangeValueLabel_)
    {
        ctTimingExposureRangeValueLabel_->setText(result.analysis.exposureTimingAvailable
            ? QStringLiteral("%1 - %2 ms")
                .arg(result.analysis.minExposureMs, 0, 'f', 6)
                .arg(result.analysis.maxExposureMs, 0, 'f', 6)
            : QStringLiteral("Unavailable"));
    }
    if (ctTimingRangeValueLabel_)
    {
        ctTimingRangeValueLabel_->setText(result.analysis.valid
            ? QStringLiteral("%1 - %2 ms")
                .arg(result.analysis.minPeriodMs, 0, 'f', 6)
                .arg(result.analysis.maxPeriodMs, 0, 'f', 6)
            : QStringLiteral("-"));
    }
    if (ctTimingFramesValueLabel_)
    {
        ctTimingFramesValueLabel_->setText(result.analysis.valid
            ? QStringLiteral("%1 - %2")
                .arg(result.analysis.firstFrameNumber)
                .arg(result.analysis.lastFrameNumber)
            : QStringLiteral("-"));
    }
    if (ctTimingDroppedValueLabel_)
    {
        ctTimingDroppedValueLabel_->setText(result.analysis.valid
            ? QString::number(result.analysis.droppedFrames)
            : QStringLiteral("-"));
    }

    cameraSnapButton_->setEnabled(!ctRunning_);
    cameraLiveButton_->setEnabled(!ctRunning_);
    cameraApplyRoiButton_->setEnabled(!ctRunning_);
    cameraResetRoiButton_->setEnabled(!ctRunning_);
    cameraExposureSpin_->setEnabled(!ctRunning_);
    cameraExposureUnitCombo_->setEnabled(!ctRunning_);
    cameraRoiXSpin_->setEnabled(!ctRunning_);
    cameraRoiYSpin_->setEnabled(!ctRunning_);
    cameraRoiWidthSpin_->setEnabled(!ctRunning_);
    cameraRoiHeightSpin_->setEnabled(!ctRunning_);
    if (ctTimingCalibrateButton_)
    {
        ctTimingCalibrateButton_->setEnabled(!ctRunning_);
        ctTimingCalibrateButton_->setText(timingCalibrationFromCache_
            ? QStringLiteral("Re-test Timing") : QStringLiteral("Calibrate Timing"));
    }
    updateCtModeUiState();
    updateCtFlyDerivedParams();
    updateCtTimeSeriesDerivedParams();

    if (timingCalibrationValid_ && !timingCalibrationFromCache_)
    {
        storeTimingProfile(result);
    }

    const bool startSmokeAcquisition = ctSmokeWaitingForTiming_;
    const bool restoreLive = timingRestoreLiveAfterFinish_ && !startSmokeAcquisition;
    timingRestoreLiveAfterFinish_ = false;
    if (restoreLive && !ctRunning_ && !cameraLiveRunning_)
    {
        QTimer::singleShot(0, this, &MainWindow::toggleCameraLivePreview);
    }
    if (startSmokeAcquisition)
    {
        ctSmokeWaitingForTiming_ = false;
        if (!timingCalibrationValid_)
        {
            appendLog(QStringLiteral("CT smoke autostart stopped: Timing calibration failed: %1")
                .arg(result.message.isEmpty() ? timingInvalidReason_ : result.message));
            return;
        }
        if (ctSmokeTargetTab_ == kCtTabTiming)
        {
            appendLog(QStringLiteral("CT smoke Timing-only calibration completed."));
            return;
        }
        if (ctModeTabs_)
        {
            ctModeTabs_->setCurrentIndex(ctSmokeTargetTab_);
        }
        ctSmokeAcquisitionStarted_ = true;
        QTimer::singleShot(0, this, &MainWindow::startCtAcquisition);
    }
}

void MainWindow::updateCtModeUiState()
{
    const int modeIndex = ctModeTabs_ ? ctModeTabs_->currentIndex() : kCtTabStep;
    const bool controlsEnabled = !ctRunning_ && !timingCalibrationRunning_;
    const bool timingReady = timingCalibrationValid_ && timingCalibrationMatchesCurrentUi();
    const bool stepMode = modeIndex == kCtTabStep;
    const bool timingMode = modeIndex == kCtTabTiming;
    const bool flyMode = modeIndex == kCtTabFly;
    const bool timeSeriesMode = modeIndex == kCtTabTimeSeries;

    if (ctModeTabs_)
    {
        ctModeTabs_->setTabEnabled(kCtTabStep, true);
        ctModeTabs_->setTabEnabled(kCtTabTiming, !ctRunning_);
        ctModeTabs_->setTabEnabled(kCtTabFly, timingReady);
        ctModeTabs_->setTabEnabled(kCtTabTimeSeries, timingReady);
    }
    if (ctStartButton_)
    {
        ctStartButton_->setEnabled(controlsEnabled
            && !timingMode
            && (stepMode || ((flyMode || timeSeriesMode) && timingReady)));
    }
    if (ctTimingCalibrateButton_)
    {
        ctTimingCalibrateButton_->setEnabled(controlsEnabled && !ctRunning_);
    }
    if (ctBeginningAngleSpin_)
    {
        ctBeginningAngleSpin_->setEnabled(controlsEnabled && (stepMode || flyMode));
    }
    if (ctStepFlatGapSpin_)
    {
        ctStepFlatGapSpin_->setEnabled(controlsEnabled && stepMode);
    }
    if (ctStepFlatMoveSpin_)
    {
        ctStepFlatMoveSpin_->setEnabled(controlsEnabled && stepMode);
    }
    if (ctStepFlatNumSpin_)
    {
        ctStepFlatNumSpin_->setEnabled(controlsEnabled && stepMode);
    }
    if (ctFlyFlatMoveSpin_)
    {
        ctFlyFlatMoveSpin_->setEnabled(controlsEnabled && flyMode);
    }
    if (ctFlyFlatNumSpin_)
    {
        ctFlyFlatNumSpin_->setEnabled(controlsEnabled && flyMode);
    }
    if (ctTimeSeriesRawFrameStrideSpin_)
    {
        ctTimeSeriesRawFrameStrideSpin_->setEnabled(controlsEnabled && timeSeriesMode);
    }
}

void MainWindow::updateCtFlyDerivedParams()
{
    if (!ctFlySpeedLabel_)
    {
        return;
    }

    PvcamCaptureSettings cameraSettings;
    cameraSettings.exposureValue = cameraExposureSpin_ ? cameraExposureSpin_->value() : 1.0;
    cameraSettings.exposureUnit = cameraExposureUnitCombo_
        ? cameraExposureUnitFromText(cameraExposureUnitCombo_->currentText())
        : PvcamExposureUnit::Milliseconds;
    const double exposureMs = exposureMsFromSettings(cameraSettings);
    const double radiusPx = static_cast<double>(cameraRoiWidthSpin_ ? cameraRoiWidthSpin_->value() : 0) * 0.5;
    const double allowedBlurPx = ctFlyAllowedBlurSpin_ ? ctFlyAllowedBlurSpin_->value() : 1.0;
    const double totalAngleDeg = ctFlyTotalAngleSpin_ ? ctFlyTotalAngleSpin_->value() : 180.0;
    const int projectionCount = ctProjectionNumSpin_ ? ctProjectionNumSpin_->value() : 1;

    QStringList problems;
    FlyTimingPlan plan;
    double leadAngleDeg = kCtFlyLeadMarginDeg;
    double firstKeepDelayS = 0.0;
    double estimatedBlurPx = 0.0;
    const bool timingReady = timingCalibrationValid_ && timingCalibrationMatchesCurrentUi();
    if (!timingReady)
    {
        problems.append(QStringLiteral("Run Timing calibration for the current Exposure and ROI."));
    }
    else
    {
        plan = makeFlyTimingPlan(totalAngleDeg, projectionCount, allowedBlurPx, radiusPx,
            exposureMs, timingCalibrationResult_.analysis.fittedPeriodMs);
        if (!plan.valid)
        {
            problems.append(plan.error);
        }
    }

    if (plan.valid)
    {
        const auto rIt = motorAxes_.constFind(QStringLiteral("R"));
        double rStartSpeed = std::min(1.0, plan.flySpeedDegPerSec);
        double rAccelTimeS = 1.5;
        if (rIt != motorAxes_.constEnd())
        {
            if (rIt->baseSpeedSpin)
            {
                rStartSpeed = std::min(rIt->baseSpeedSpin->value(), plan.flySpeedDegPerSec * 0.5);
            }
            if (rIt->accelSpin)
            {
                rAccelTimeS = std::max(0.01, rIt->accelSpin->value());
            }
        }
        const double accelDistanceDeg = 0.5 * (std::max(0.0, rStartSpeed) + plan.flySpeedDegPerSec) * rAccelTimeS;
        leadAngleDeg = std::min(90.0, std::max(kCtFlyLeadMarginDeg, accelDistanceDeg + kCtFlyLeadMarginDeg));
        firstKeepDelayS = rAccelTimeS
            + std::max(0.0, leadAngleDeg - accelDistanceDeg) / std::max(1e-9, plan.flySpeedDegPerSec);
        const double radToDeg = 180.0 / std::acos(-1.0);
        estimatedBlurPx = std::tan((plan.flySpeedDegPerSec * exposureMs / 1000.0) / radToDeg) * radiusPx;
        if (rIt != motorAxes_.constEnd() && rIt->speedSpin && plan.flySpeedDegPerSec > rIt->speedSpin->value())
        {
            problems.append(QStringLiteral("Derived speed exceeds current R Top Speed/s (%1 deg/s).")
                .arg(rIt->speedSpin->value(), 0, 'f', 4));
        }
        if (leadAngleDeg >= 90.0)
        {
            problems.append(QStringLiteral("Derived lead reached the 90 deg safety clamp."));
        }
    }

    ctFlyRawFrameStride_ = plan.valid ? plan.rawFrameStride : 1;
    ctFlyDerivedSpeedDegPerSec_ = plan.valid ? plan.flySpeedDegPerSec : 0.0;
    ctFlyDerivedLeadAngleDeg_ = leadAngleDeg;
    ctFlyDerivedScanTimeS_ = plan.valid ? plan.scanTimeS : 0.0;
    ctFlyDerivedFrameIntervalMs_ = plan.valid ? plan.projectionIntervalMs : 0.0;
    ctFlyFirstKeepDelayMs_ = firstKeepDelayS * 1000.0;
    ctFlyRadiusLabel_->setText(radiusPx > 0.0 ? QStringLiteral("%1 px").arg(radiusPx, 0, 'f', 1) : QStringLiteral("-"));
    ctFlySpeedLabel_->setText(plan.valid ? QString::number(plan.flySpeedDegPerSec, 'f', 6) : QStringLiteral("-"));
    ctFlyLeadAngleLabel_->setText(plan.valid
        ? QStringLiteral("%1 (margin %2)").arg(leadAngleDeg, 0, 'f', 4).arg(kCtFlyLeadMarginDeg, 0, 'f', 1)
        : QStringLiteral("-"));
    ctFlyScanTimeLabel_->setText(plan.valid ? formatDurationSeconds(plan.scanTimeS) : QStringLiteral("-"));
    ctFlyFrameIntervalLabel_->setText(plan.valid
        ? QStringLiteral("%1 ms").arg(plan.projectionIntervalMs, 0, 'f', 6)
        : QStringLiteral("-"));
    if (ctFlyRawStrideLabel_)
    {
        ctFlyRawStrideLabel_->setText(plan.valid
            ? QStringLiteral("Every %1 raw frame(s)").arg(plan.rawFrameStride)
            : QStringLiteral("-"));
    }
    if (ctFlyValidationLabel_)
    {
        if (problems.isEmpty() && plan.valid)
        {
            ctFlyValidationLabel_->setText(QStringLiteral("OK: raw %1 ms, stride %2, blur %3 px, first keep delay %4 s.")
                .arg(timingCalibrationResult_.analysis.fittedPeriodMs, 0, 'f', 6)
                .arg(plan.rawFrameStride)
                .arg(estimatedBlurPx, 0, 'f', 3)
                .arg(firstKeepDelayS, 0, 'f', 3));
            ctFlyValidationLabel_->setStyleSheet(QStringLiteral("color:#1b7f3a;"));
        }
        else
        {
            ctFlyValidationLabel_->setText(problems.join(QStringLiteral(" ")));
            ctFlyValidationLabel_->setStyleSheet(QStringLiteral("color:#b00020;"));
        }
    }
}

void MainWindow::updateCtTimeSeriesDerivedParams()
{
    if (!ctTimeSeriesRawFrameStrideSpin_)
    {
        return;
    }
    const bool timingReady = timingCalibrationValid_ && timingCalibrationMatchesCurrentUi();
    TimeSeriesTimingPlan plan;
    if (timingReady)
    {
        plan = makeTimeSeriesTimingPlan(ctTimeSeriesRawFrameStrideSpin_->value(),
            timingCalibrationResult_.analysis.fittedPeriodMs);
    }
    ctTimeSeriesRawFrameStride_ = plan.valid ? plan.rawFrameStride : 1;
    ctTimeSeriesActualIntervalMs_ = plan.valid ? plan.actualIntervalMs : 0.0;
    const int frameCount = ctProjectionNumSpin_ ? ctProjectionNumSpin_->value() : 0;
    PvcamCaptureSettings cameraSettings;
    cameraSettings.exposureValue = cameraExposureSpin_ ? cameraExposureSpin_->value() : 1.0;
    cameraSettings.exposureUnit = cameraExposureUnitCombo_
        ? cameraExposureUnitFromText(cameraExposureUnitCombo_->currentText())
        : PvcamExposureUnit::Milliseconds;
    const double exposureMs = exposureMsFromSettings(cameraSettings);
    const double durationMs = plan.valid && frameCount > 0
        ? static_cast<double>(std::max(0, frameCount - 1)) * plan.actualIntervalMs + exposureMs
        : 0.0;
    const qint64 frameBytes = static_cast<qint64>(cameraRoiWidthSpin_ ? cameraRoiWidthSpin_->value() : 0)
        * static_cast<qint64>(cameraRoiHeightSpin_ ? cameraRoiHeightSpin_->value() : 0) * 2;
    const double dataGiB = static_cast<double>(std::max(0, frameCount)) * static_cast<double>(frameBytes)
        / (1024.0 * 1024.0 * 1024.0);

    if (ctTimeSeriesActualIntervalLabel_)
    {
        ctTimeSeriesActualIntervalLabel_->setText(plan.valid
            ? QStringLiteral("%1 ms").arg(plan.actualIntervalMs, 0, 'f', 6)
            : QStringLiteral("-"));
    }
    if (ctTimeSeriesActualRateLabel_)
    {
        ctTimeSeriesActualRateLabel_->setText(plan.valid
            ? QStringLiteral("%1 fps").arg(plan.actualFrameRateFps, 0, 'f', 3)
            : QStringLiteral("-"));
    }
    if (ctTimeSeriesDurationLabel_)
    {
        ctTimeSeriesDurationLabel_->setText(plan.valid ? formatDurationSeconds(durationMs / 1000.0) : QStringLiteral("-"));
    }
    if (ctTimeSeriesDataSizeLabel_)
    {
        ctTimeSeriesDataSizeLabel_->setText(plan.valid ? QStringLiteral("%1 GiB").arg(dataGiB, 0, 'f', 2) : QStringLiteral("-"));
    }
    if (ctTimeSeriesValidationLabel_)
    {
        ctTimeSeriesValidationLabel_->setText(plan.valid
            ? QStringLiteral("OK: save FrameNr F0 + k x %1; skip %2 raw frame(s) between saved frames; actual interval %3 ms.")
                .arg(plan.rawFrameStride)
                .arg(std::max(0, plan.rawFrameStride - 1))
                .arg(plan.actualIntervalMs, 0, 'f', 6)
            : QStringLiteral("Run Timing calibration for the current Exposure and ROI."));
        ctTimeSeriesValidationLabel_->setStyleSheet(plan.valid
            ? QStringLiteral("color:#1b7f3a;")
            : QStringLiteral("color:#b00020;"));
    }
}

void MainWindow::updateContinuousTimingUi()
{
    if (!ctFrameTimingLabel_)
    {
        return;
    }
    const int selectedMode = ctModeTabs_ ? ctModeTabs_->currentIndex() : kCtTabStep;
    const bool continuousMode = ctRunning_
        ? (ctSettings_.mode == CtAcquisitionMode::FlyScan
            || ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
        : (selectedMode == kCtTabFly || selectedMode == kCtTabTimeSeries);
    if (!continuousMode)
    {
        ctFrameTimingLabel_->setText(QStringLiteral("-"));
        ctFrameTimingLabel_->setStyleSheet(QString());
        return;
    }

    const ContinuousTimingStats& stats = ctContinuousTimingStats_;
    if (!stats.valid && !stats.frameLossDetected && !stats.invalidTimestampDetected)
    {
        ctFrameTimingLabel_->setText(QStringLiteral("Waiting for hardware timestamps..."));
        ctFrameTimingLabel_->setStyleSheet(QStringLiteral("color:#555555;"));
        return;
    }

    const double calibratedFps = stats.baselinePeriodMs > 0.0
        ? 1000.0 / stats.baselinePeriodMs
        : 0.0;
    const double maxAngleErrorDeg = ctSettings_.mode == CtAcquisitionMode::FlyScan
        ? ctSettings_.flySpeedDegPerSec * stats.maxAbsPhaseErrorMs / 1000.0
        : 0.0;
    QString secondLine = QStringLiteral("Max interval %1 ms | Phase %2 / max %3 ms | Duplicates %4 | Drops %5")
        .arg(stats.maxIntervalDeviationMs, 0, 'f', 4)
        .arg(stats.currentPhaseErrorMs, 0, 'f', 4)
        .arg(stats.maxAbsPhaseErrorMs, 0, 'f', 4)
        .arg(stats.duplicateNotifications)
        .arg(stats.deviceDroppedFrames);
    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        secondLine += QStringLiteral(" | Max angle %1 deg").arg(maxAngleErrorDeg, 0, 'f', 6);
    }
    ctFrameTimingLabel_->setText(QStringLiteral("Cal %1 fps | Run %2 fps | Drift %3%\n%4")
        .arg(calibratedFps, 0, 'f', 3)
        .arg(stats.rollingFrameRateFps, 0, 'f', 3)
        .arg(stats.rollingDriftPercent, 0, 'f', 3)
        .arg(secondLine));

    if (stats.frameLossDetected || stats.invalidTimestampDetected || stats.keptStrideViolationDetected)
    {
        ctFrameTimingLabel_->setStyleSheet(QStringLiteral("color:#b00020; font-weight:600;"));
    }
    else if (std::abs(stats.rollingDriftPercent) > 1.0)
    {
        ctFrameTimingLabel_->setStyleSheet(QStringLiteral("color:#b00020;"));
    }
    else if (std::abs(stats.rollingDriftPercent) > 0.2)
    {
        ctFrameTimingLabel_->setStyleSheet(QStringLiteral("color:#9a6700;"));
    }
    else
    {
        ctFrameTimingLabel_->setStyleSheet(QStringLiteral("color:#1b7f3a;"));
    }
}

int MainWindow::estimateCtTotalFrames() const
{
    const int projections = std::max(0, ctSettings_.projectionCount);
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        return projections;
    }
    int frames = projections;
    if (ctSettings_.autoCor)
    {
        frames += 1;
    }
    if (ctSettings_.flatCount > 0)
    {
        int flatGroups = 2;
        if (ctSettings_.mode == CtAcquisitionMode::StepAndShoot
            && ctSettings_.flatGap > 0
            && projections > 1)
        {
            flatGroups += (projections - 1) / ctSettings_.flatGap;
        }
        frames += flatGroups * ctSettings_.flatCount;
    }
    return std::max(0, frames);
}

double MainWindow::estimateAxisMoveSeconds(const QString& axisName, double distanceUnits) const
{
    const auto it = motorAxes_.find(axisName);
    if (it == motorAxes_.end())
    {
        return 0.0;
    }
    const double distance = std::abs(distanceUnits);
    const double topSpeed = it->speedSpin ? std::max(1e-9, it->speedSpin->value()) : 1.0;
    const double accelTime = it->accelSpin ? std::max(0.0, it->accelSpin->value()) : 0.0;
    constexpr double kCommandSettleSeconds = 0.08;
    return distance / topSpeed + accelTime + kCommandSettleSeconds;
}

double MainWindow::estimateCtAcquisitionSeconds(int totalFrames) const
{
    if (totalFrames <= 0)
    {
        return 0.0;
    }
    const int projections = std::max(0, ctSettings_.projectionCount);
    const double exposureS = std::max(0.001, exposureMsFromSettings(ctSettings_.camera)) / 1000.0;
    constexpr double kPreparedFrameOverheadS = 0.04;
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        const double intervalS = std::max(0.001, ctSettings_.timeSeriesFrameIntervalMs) / 1000.0;
        return projections > 0
            ? (static_cast<double>(std::max(0, projections - 1)) * intervalS + exposureS)
            : 0.0;
    }
    const int corFrames = ctSettings_.autoCor ? 1 : 0;
    const int flatFrames = std::max(0, totalFrames - projections - corFrames);

    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        const double scanS = ctSettings_.flyScanTimeS > 0.0
            ? ctSettings_.flyScanTimeS
            : (std::max(0.0, ctSettings_.flyFrameIntervalMs) * static_cast<double>(projections) / 1000.0);
        const double referenceFrameS = static_cast<double>(flatFrames) * (exposureS + kPreparedFrameOverheadS);
        const int flatGroups = ctSettings_.flatCount > 0 ? 2 : 0;
        const double yMoveS = static_cast<double>(flatGroups) * 2.0
            * estimateAxisMoveSeconds(QStringLiteral("Y"), ctSettings_.flatMoveDistanceUm);
        const double flySweepS = ctSettings_.flySpeedDegPerSec > 0.0
            ? ((180.0 + 2.0 * std::max(0.0, ctSettings_.flyLeadAngleDeg)) / ctSettings_.flySpeedDegPerSec
                + estimateAxisMoveSeconds(QStringLiteral("R"), 0.0))
            : scanS;
        return std::max(scanS, flySweepS) + referenceFrameS + yMoveS;
    }

    int flatGroups = 0;
    if (ctSettings_.flatCount > 0)
    {
        flatGroups = 2;
        if (ctSettings_.flatGap > 0 && projections > 1)
        {
            flatGroups += (projections - 1) / ctSettings_.flatGap;
        }
    }
    const double angleStep = projections > 0 ? 180.0 / static_cast<double>(projections) : 0.0;
    const int smallRMoveCount = projections;
    const double rSmallMoveS = static_cast<double>(smallRMoveCount)
        * estimateAxisMoveSeconds(QStringLiteral("R"), angleStep);
    const double rInitialMoveS = estimateAxisMoveSeconds(QStringLiteral("R"), 0.0);
    const double yMoveS = static_cast<double>(flatGroups) * 2.0
        * estimateAxisMoveSeconds(QStringLiteral("Y"), ctSettings_.flatMoveDistanceUm);
    const double cameraS = static_cast<double>(totalFrames) * (exposureS + kPreparedFrameOverheadS);
    return cameraS + rInitialMoveS + rSmallMoveS + yMoveS;
}

void MainWindow::updateCtProgressUi()
{
    if (!ctProgressBar_ || !ctTimingLabel_)
    {
        return;
    }

    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    const double elapsedS = ctAcquisitionStartedMs_ > 0
        ? static_cast<double>(std::max<qint64>(0, (ctCompletedMs_ > 0 && !ctRunning_) ? (ctCompletedMs_ - ctAcquisitionStartedMs_) : (nowMs - ctAcquisitionStartedMs_))) / 1000.0
        : 0.0;
    const int totalFrames = std::max(0, ctEstimatedTotalFrames_);
    const int savedFrames = std::max(0, ctFrameSavedCount_);
    const double frameRatio = totalFrames > 0
        ? std::clamp(static_cast<double>(savedFrames) / static_cast<double>(totalFrames), 0.0, 1.0)
        : 0.0;
    const double timeRatio = ctEstimatedTotalSeconds_ > 0.0
        ? std::clamp(elapsedS / ctEstimatedTotalSeconds_, 0.0, 1.0)
        : 0.0;
    double progressRatio = std::max(frameRatio, timeRatio);
    if (ctRunning_)
    {
        progressRatio = std::min(progressRatio, 0.99);
    }

    ctProgressBar_->setValue(static_cast<int>(std::llround(progressRatio * static_cast<double>(ctProgressBar_->maximum()))));

    double remainingS = 0.0;
    const QString frameText = totalFrames > 0
        ? QStringLiteral("Frames %1/%2").arg(savedFrames).arg(totalFrames)
        : QStringLiteral("Frames %1").arg(savedFrames);
    if (ctRunning_)
    {
        if (savedFrames >= 3 && totalFrames > savedFrames)
        {
            remainingS = (elapsedS / static_cast<double>(savedFrames)) * static_cast<double>(totalFrames - savedFrames);
        }
        else if (ctEstimatedTotalSeconds_ > 0.0)
        {
            remainingS = std::max(0.0, ctEstimatedTotalSeconds_ - elapsedS);
        }
        ctTimingLabel_->setText(QStringLiteral("%1 | Elapsed %2 / remaining %3")
            .arg(frameText, formatClockDuration(elapsedS), formatClockDuration(remainingS)));
    }
    else if (ctCompletedMs_ > 0)
    {
        ctTimingLabel_->setText(QStringLiteral("%1 | Elapsed %2 / remaining 0:00")
            .arg(frameText, formatClockDuration(elapsedS)));
    }
    else
    {
        ctTimingLabel_->setText(QStringLiteral("Elapsed 0:00 / remaining --"));
    }
}

void MainWindow::appendLog(const QString& text)
{
    if (isRoutineMotorLog(text))
    {
        return;
    }
    const QDateTime now = QDateTime::currentDateTime();
    const QString uiText = QStringLiteral("[%1] %2").arg(now.toString(QStringLiteral("HH:mm:ss.zzz")), text);
    logEdit_->append(uiText);
    recentLogLines_.append(uiText);
    while (recentLogLines_.size() > 200)
    {
        recentLogLines_.removeFirst();
    }

    QFile file(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("ct_workstation_run.log")));
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        QTextStream stream(&file);
        stream << now.toString(Qt::ISODateWithMs) << " " << text << '\n';
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
        if (!timingCalibrationRunning_
            && (state.contains(QStringLiteral("Error"), Qt::CaseInsensitive)
                || state.contains(QStringLiteral("unavailable"), Qt::CaseInsensitive)))
        {
            invalidateTimingCalibration(QStringLiteral("Camera session is not ready."));
        }
        if (!cameraWarmOpenComplete_
            && (state.contains(QStringLiteral("Ready"), Qt::CaseInsensitive)
                || state.contains(QStringLiteral("Error"), Qt::CaseInsensitive)
                || state.contains(QStringLiteral("unavailable"), Qt::CaseInsensitive)))
        {
            cameraWarmOpenComplete_ = true;
            cameraWarmOpenCompletedMs_ = QDateTime::currentMSecsSinceEpoch();
            writeStartupTrace(QStringLiteral("camera warm open complete: %1").arg(state));
            if (reconStartDeferredForCameraWarmOpen_)
            {
                reconStartDeferredForCameraWarmOpen_ = false;
                QTimer::singleShot(static_cast<int>(kReconQueueCameraSettleMs), this, &MainWindow::startNextQueuedReconstruction);
            }
        }
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
    connect(cameraWorker_, &PvcamCameraWorker::timingCalibrationFinished,
        this, &MainWindow::handleTimingCalibrationResult);
    connect(cameraWorker_, &PvcamCameraWorker::continuousStatsChanged, this,
        [this](qint64 rawFrames, qint64 keptFrames, qint64 discardedFrames,
            qint64 deviceDroppedFrames, int poolInUse, int poolPeak) {
            ctContinuousRawFrames_ = rawFrames;
            ctContinuousDiscardedFrames_ = discardedFrames;
            ctContinuousDeviceDroppedFrames_ = deviceDroppedFrames;
            ctFramePoolInUse_ = poolInUse;
            ctFramePoolPeak_ = std::max(ctFramePoolPeak_, poolPeak);
            if (ctRunning_ && (ctSettings_.mode == CtAcquisitionMode::FlyScan
                || ctSettings_.mode == CtAcquisitionMode::TimeSeries2D))
            {
                ctFrameSavedCount_ = ctContinuousFrameBaseCount_ + static_cast<int>(keptFrames);
                if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
                {
                    ctProjectionIndex_ = static_cast<int>(keptFrames);
                    ctStatusLabel_->setText(QStringLiteral("2D %1/%2")
                        .arg(std::min(ctProjectionIndex_, ctSettings_.projectionCount))
                        .arg(ctSettings_.projectionCount));
                }
                else
                {
                    ctStatusLabel_->setText(QStringLiteral("Fly Tomo %1/%2")
                        .arg(std::min(static_cast<int>(keptFrames), ctSettings_.projectionCount))
                        .arg(ctSettings_.projectionCount));
                }
                updateCtProgressUi();
            }
            if (ctBufferLabel_)
            {
                const double inUseGiB = static_cast<double>(poolInUse) * static_cast<double>(ctWriterFrameBytes_)
                    / (1024.0 * 1024.0 * 1024.0);
                const double capacityGiB = static_cast<double>(kCtFramePoolCapacity) * static_cast<double>(ctWriterFrameBytes_)
                    / (1024.0 * 1024.0 * 1024.0);
                ctBufferLabel_->setText(QStringLiteral("%1 / %2 frames, peak %3, %4 / %5 GiB")
                    .arg(poolInUse)
                    .arg(kCtFramePoolCapacity)
                    .arg(ctFramePoolPeak_)
                    .arg(inUseGiB, 0, 'f', 1)
                    .arg(capacityGiB, 0, 'f', 1));
            }
        });
    connect(cameraWorker_, &PvcamCameraWorker::continuousRingStatsChanged, this,
        [this](int estimatedUnread, int capacity, int peakEstimatedUnread) {
            ctPvcamRingEstimatedUnread_ = estimatedUnread;
            ctPvcamRingCapacity_ = capacity;
            ctPvcamRingPeakEstimatedUnread_ = std::max(ctPvcamRingPeakEstimatedUnread_, peakEstimatedUnread);
            const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
            if (ctPvcamRingLabel_ && (ctPvcamRingLastUiUpdateMs_ == 0
                || nowMs - ctPvcamRingLastUiUpdateMs_ >= 100))
            {
                ctPvcamRingLastUiUpdateMs_ = nowMs;
                ctPvcamRingLabel_->setText(QStringLiteral("estimated unread %1 / %2 | peak %3")
                    .arg(estimatedUnread)
                    .arg(capacity)
                    .arg(ctPvcamRingPeakEstimatedUnread_));
            }
        });
    connect(cameraWorker_, &PvcamCameraWorker::continuousTimingStatsChanged, this,
        [this](const ContinuousTimingStats& stats) {
            if (!ctRunning_ || (ctSettings_.mode != CtAcquisitionMode::FlyScan
                && ctSettings_.mode != CtAcquisitionMode::TimeSeries2D))
            {
                return;
            }
            ctContinuousTimingStats_ = stats;
            ctContinuousDeviceDroppedFrames_ = stats.deviceDroppedFrames;
            if (stats.valid)
            {
                const double absDrift = std::abs(stats.rollingDriftPercent);
                const int driftLevel = absDrift > 1.0 ? 2 : (absDrift > 0.2 ? 1 : 0);
                if (driftLevel != ctContinuousTimingDriftLevel_)
                {
                    if (driftLevel > 0)
                    {
                        appendLog(QStringLiteral("Continuous timing warning: rolling drift=%1%, rolling=%2 ms, baseline=%3 ms.")
                            .arg(stats.rollingDriftPercent, 0, 'f', 4)
                            .arg(stats.rollingPeriodMs, 0, 'f', 6)
                            .arg(stats.baselinePeriodMs, 0, 'f', 6));
                    }
                    else if (ctContinuousTimingDriftLevel_ > 0)
                    {
                        appendLog(QStringLiteral("Continuous timing recovered: rolling drift=%1%.")
                            .arg(stats.rollingDriftPercent, 0, 'f', 4));
                    }
                    ctContinuousTimingDriftLevel_ = driftLevel;
                }
            }
            updateContinuousTimingUi();
            if (stats.frameLossDetected && ctStatusLabel_)
            {
                ctStatusLabel_->setText(QStringLiteral("Failed: device frame lost"));
                ctStatusLabel_->setStyleSheet(QStringLiteral("color:#b00020; font-weight:600;"));
            }
        });
    connect(cameraWorker_, &PvcamCameraWorker::continuousIntervalStrategyResolved, this, [this](const QString& strategy, const QString& reason) {
        if (!ctRunning_)
        {
            return;
        }
        ctContinuousIntervalStrategy_ = strategy;
        ctContinuousIntervalFallbackReason_ = reason;
    });
    connect(cameraWorker_, &PvcamCameraWorker::captureFinished, this, [this](bool ok, const QString& message) {
        if (shuttingDown_) { return; }
        if (ctRunning_)
        {
            if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D && ctTimeSeriesCapturing_)
            {
                if (!ctContinuousCaptureActive_)
                {
                    appendLog(QStringLiteral("Ignoring stale camera completion while 2D continuous capture is starting: %1")
                        .arg(message));
                    return;
                }
                ctContinuousCaptureActive_ = false;
                ctContinuousStartPending_ = false;
                ctTimeSeriesCapturing_ = false;
                ctWaitingForFrame_ = false;
                ctWaitingForCaptureFinish_ = false;
                ctFlyScanFinishedMs_ = QDateTime::currentMSecsSinceEpoch();
                if (ctCurrentFrameBatchStartedMs_ > 0)
                {
                    ctFrameCaptureMs_ += std::max<qint64>(0, ctFlyScanFinishedMs_ - ctCurrentFrameBatchStartedMs_);
                    ctCurrentFrameBatchStartedMs_ = 0;
                    ++ctFrameBatchCount_;
                }
                appendLog(QStringLiteral("Kinetix 2D time series capture %1: %2")
                    .arg(ok ? QStringLiteral("ok") : QStringLiteral("failed"), message));
                if (!ok && !ctStopRequested_)
                {
                    ctPendingFinishErrorMessage_ = QStringLiteral("Kinetix 2D time series failed: %1").arg(message);
                }
                if (ctWriter_ && ctWriterFinishRequestedMs_ <= 0)
                {
                    ctWriterFinishRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
                    QMetaObject::invokeMethod(ctWriter_, "finish", Qt::QueuedConnection);
                }
                return;
            }
            if (ctSettings_.mode == CtAcquisitionMode::FlyScan && ctFlyScanCapturing_)
            {
                if (!ctContinuousCaptureActive_)
                {
                    appendLog(QStringLiteral("Ignoring stale camera completion while Fly continuous capture is starting: %1")
                        .arg(message));
                    return;
                }
                ctContinuousCaptureActive_ = false;
                ctContinuousStartPending_ = false;
                ctFlyScanCapturing_ = false;
                ctWaitingForFrame_ = false;
                ctWaitingForCaptureFinish_ = false;
                ctFlyScanFinishedMs_ = QDateTime::currentMSecsSinceEpoch();
                if (ctCurrentFrameBatchStartedMs_ > 0)
                {
                    ctFrameCaptureMs_ += std::max<qint64>(0, ctFlyScanFinishedMs_ - ctCurrentFrameBatchStartedMs_);
                    ctCurrentFrameBatchStartedMs_ = 0;
                    ++ctFrameBatchCount_;
                }
                appendLog(QStringLiteral("Kinetix fly scan capture %1: %2")
                    .arg(ok ? QStringLiteral("ok") : QStringLiteral("failed"), message));
                if (!ok)
                {
                    ctPendingFinishErrorMessage_ = QStringLiteral("Kinetix fly scan failed: %1").arg(message);
                    if (!ctStopRequested_ && kohzuController_)
                    {
                        QMetaObject::invokeMethod(kohzuController_, "stopAllAxes", Qt::QueuedConnection);
                    }
                    if (ctWriter_ && ctWriterFinishRequestedMs_ <= 0)
                    {
                        ctWriterFinishRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
                        QMetaObject::invokeMethod(ctWriter_, "finish", Qt::QueuedConnection);
                    }
                    return;
                }
                ctFlyRestartingPreparedAfterScan_ = true;
                ctPreparedStartedMs_ = QDateTime::currentMSecsSinceEpoch();
                QMetaObject::invokeMethod(cameraWorker_, "beginPreparedCapture", Qt::QueuedConnection,
                    Q_ARG(ct::PvcamCaptureSettings, ctSettings_.camera));
                return;
            }
            if (!ok && !ctReplayRunning_)
            {
                finishCtAcquisition(false, QStringLiteral("Kinetix capture failed outside prepared CT path: %1").arg(message));
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
            cameraSnapIndexSpin_->setValue(std::min(cameraSnapIndexSpin_->maximum(), completedSingleSnapIndex + 1));
        }
        appendLog(QStringLiteral("Kinetix capture %1: %2").arg(ok ? QStringLiteral("ok") : QStringLiteral("failed"), message));
        if (timingCalibrationPendingAfterLiveStop_)
        {
            timingCalibrationPendingAfterLiveStop_ = false;
            if (ok)
            {
                QTimer::singleShot(0, this, &MainWindow::startTimingCalibration);
            }
            else
            {
                timingRestoreLiveAfterFinish_ = false;
                invalidateTimingCalibration(QStringLiteral("Failed to stop Live before timing calibration."));
            }
            return;
        }
        if (shouldRestartLive && ok)
        {
            toggleCameraLivePreview();
        }
    });
    connect(cameraWorker_, &PvcamCameraWorker::preparedCaptureReady, this, [this](bool ok, const QString& message) {
        if (shuttingDown_ || !ctRunning_ || ctReplayRunning_) { return; }
        if (!ok)
        {
            finishCtAcquisition(false, QStringLiteral("Kinetix CT prepared setup failed: %1").arg(message));
            return;
        }
        ctPreparedReadyMs_ = QDateTime::currentMSecsSinceEpoch();
        if (ctPreparedStartedMs_ > 0)
        {
            ctPreparedSetupMs_ += std::max<qint64>(0, ctPreparedReadyMs_ - ctPreparedStartedMs_);
        }
        appendLog(message);
        if (ctSettings_.mode == CtAcquisitionMode::FlyScan && ctFlyRestartingPreparedAfterScan_)
        {
            ctFlyRestartingPreparedAfterScan_ = false;
            QTimer::singleShot(0, this, &MainWindow::advanceCtFlyAcquisition);
            return;
        }
        if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
        {
            startCtFlyInitialMove();
        }
        else
        {
            startCtInitialMotorMove();
        }
    });
    connect(cameraWorker_, &PvcamCameraWorker::preparedFrameBatchFinished, this, [this](bool ok, const QString& message) {
        if (shuttingDown_ || !ctRunning_ || ctReplayRunning_) { return; }
        if (!ok)
        {
            finishCtAcquisition(false, QStringLiteral("Kinetix CT prepared capture failed: %1").arg(message));
            return;
        }
        if (ctCurrentFrameBatchStartedMs_ > 0)
        {
            ctFrameCaptureMs_ += std::max<qint64>(0, QDateTime::currentMSecsSinceEpoch() - ctCurrentFrameBatchStartedMs_);
            ctCurrentFrameBatchStartedMs_ = 0;
            ++ctFrameBatchCount_;
        }
        ctWaitingForCaptureFinish_ = false;
        if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
        {
            QTimer::singleShot(0, this, &MainWindow::advanceCtFlyAcquisition);
        }
        else
        {
            QTimer::singleShot(0, this, &MainWindow::advanceCtAcquisition);
        }
    });
    connect(cameraWorker_, &PvcamCameraWorker::continuousCaptureStarted, this, [this](bool ok, const QString& message) {
        if (shuttingDown_ || !ctRunning_)
        {
            return;
        }
        if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D && ctTimeSeriesCapturing_)
        {
            if (!ctContinuousStartPending_)
            {
                appendLog(QStringLiteral("Ignoring stale 2D continuous start notification: %1").arg(message));
                return;
            }
            ctContinuousStartPending_ = false;
            ctContinuousCaptureActive_ = ok;
            appendLog(message);
            if (!ok)
            {
                ctPendingFinishErrorMessage_ = QStringLiteral("Kinetix 2D continuous start failed: %1").arg(message);
                if (ctWriter_ && ctWriterFinishRequestedMs_ <= 0)
                {
                    ctWriterFinishRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
                    QMetaObject::invokeMethod(ctWriter_, "finish", Qt::QueuedConnection);
                }
            }
            return;
        }
        if (ctSettings_.mode != CtAcquisitionMode::FlyScan || !ctFlyScanCapturing_)
        {
            return;
        }
        if (!ctContinuousStartPending_)
        {
            appendLog(QStringLiteral("Ignoring stale Fly continuous start notification: %1").arg(message));
            return;
        }
        ctContinuousStartPending_ = false;
        ctContinuousCaptureActive_ = ok;
        appendLog(message);
        if (!ok)
        {
            finishCtAcquisition(false, QStringLiteral("Kinetix fly continuous start failed: %1").arg(message));
            return;
        }
        startCtFlyScanMotorSweep();
    });
    cameraThread_.start();
    // Continuous PVCAM polling must win a scheduling race against TIFF
    // compression. The writer is bounded and non-blocking; letting the camera
    // thread run at high priority prevents a transient CPU burst from filling
    // the hardware ring and turning into a frame-number gap.
    cameraThread_.setPriority(QThread::HighPriority);
    cameraWarmOpenComplete_ = true;
    cameraWarmOpenCompletedMs_ = QDateTime::currentMSecsSinceEpoch();
    appendLog(QStringLiteral("Kinetix warm open deferred; camera will open on Live/Snap/CT."));
    writeStartupTrace(QStringLiteral("PVCAM warmOpen deferred until first camera use"));

    ctWriter_ = new CtTiffWriter();
    ctWriter_->moveToThread(&ctWriterThread_);
    ctContinuousRouter_ = new ContinuousFrameRouter(this);
    connect(ctContinuousRouter_, &ContinuousFrameRouter::previewFrameReady, this,
        [this](const CameraFrame& frame) {
            if (ctRunning_ && (ctSettings_.mode == CtAcquisitionMode::FlyScan
                || ctSettings_.mode == CtAcquisitionMode::TimeSeries2D))
            {
                updateCameraFrameForCtPreview(frame);
            }
        }, Qt::QueuedConnection);
    connect(ctContinuousRouter_, &ContinuousFrameRouter::frameBatchFinished, this,
        &MainWindow::handleContinuousFrameBatch, Qt::QueuedConnection);
    connect(ctWriter_, &CtTiffWriter::logMessage, this, &MainWindow::appendLog);
    connect(ctWriter_, &CtTiffWriter::queueChanged, this, [this](int queued, int saved) {
        const double elapsedSeconds = ctWriterStartedMs_ > 0
            ? std::max(0.001, static_cast<double>(QDateTime::currentMSecsSinceEpoch() - ctWriterStartedMs_) / 1000.0)
            : 0.001;
        const double mibPerSecond = (static_cast<double>(saved) * static_cast<double>(ctWriterFrameBytes_))
            / (1024.0 * 1024.0) / elapsedSeconds;
        const double queuedGiB = (static_cast<double>(queued) * static_cast<double>(ctWriterFrameBytes_))
            / (1024.0 * 1024.0 * 1024.0);
        const double capacityGiB = (static_cast<double>(ctWriterQueueCapacity_) * static_cast<double>(ctWriterFrameBytes_))
            / (1024.0 * 1024.0 * 1024.0);
        if (ctWriterLabel_)
        {
            ctWriterLabel_->setText(QStringLiteral("%1 threads, %2 MiB/s, queue %3, saved %4")
                .arg(std::max(0, ctWriterThreads_))
                .arg(mibPerSecond, 0, 'f', 1)
                .arg(queued)
                .arg(saved));
        }
        if (ctBufferLabel_ && ctSettings_.mode == CtAcquisitionMode::StepAndShoot)
        {
            ctBufferLabel_->setText(QStringLiteral("%1 / %2 frames, %3 / %4 GiB")
                .arg(queued)
                .arg(std::max(0, ctWriterQueueCapacity_))
                .arg(queuedGiB, 0, 'f', 1)
                .arg(capacityGiB, 0, 'f', 1));
        }
    });
    connect(ctWriter_, &CtTiffWriter::finished, this, [this](bool ok, const QString& message) {
        appendLog(message);
        if (ctRunning_)
        {
            const bool hasCaptureError = !ctPendingFinishErrorMessage_.isEmpty();
            finishCtAcquisition(ok && !ctStopRequested_ && !hasCaptureError,
                ctStopRequested_
                    ? QStringLiteral("CT acquisition stopped by user.")
                    : (hasCaptureError ? ctPendingFinishErrorMessage_ : message));
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
            QString writeError;
            const bool ok = writeImageFile(framePath, frameToSave, &writeError);
            QMetaObject::invokeMethod(this, [this, ok, writeError, framePath, snapIndex]() {
                if (shuttingDown_) { return; }
                cameraSnapSaveInProgress_ = false;
                cameraSnapButton_->setEnabled(true);
                if (ok)
                {
                    cameraSnapIndexSpin_->setValue(std::min(cameraSnapIndexSpin_->maximum(), snapIndex + 1));
                }
                appendLog(ok
                    ? QStringLiteral("Kinetix live snap saved: %1.").arg(framePath)
                    : QStringLiteral("Kinetix snap failed: could not save %1: %2.").arg(framePath, writeError));
            }, Qt::QueuedConnection);
        }).detach();
        return;
    }

    lastRawCameraFrame_.release();
    lastCameraSaveFrame_.release();
    lastCameraHoverFrame_.release();
    cameraPreviewScaleX_ = 1.0;
    cameraPreviewScaleY_ = 1.0;
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;
    lastCameraHoverFrame_.release();
    cameraPreviewScaleX_ = 1.0;
    cameraPreviewScaleY_ = 1.0;
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
    ++liveRestoreGeneration_;
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
    lastCameraHoverFrame_.release();
    cameraPreviewScaleX_ = 1.0;
    cameraPreviewScaleY_ = 1.0;
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;
    lastCameraHoverFrame_.release();
    cameraPreviewScaleX_ = 1.0;
    cameraPreviewScaleY_ = 1.0;
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
    ++liveRestoreGeneration_;
    if (!cameraWorker_)
    {
        return;
    }
    const bool wasLiveRunning = cameraLiveRunning_;
    cameraRestartLiveAfterStop_ = restartAfterStop;
    cameraLiveRunning_ = false;
    cameraLiveButton_->setText(QStringLiteral("Open Live"));
    cameraLiveButton_->setEnabled(false);
    cameraSnapButton_->setEnabled(false);
    cameraApplyRoiButton_->setEnabled(false);
    cameraResetRoiButton_->setEnabled(false);
    if (wasLiveRunning)
    {
        QMetaObject::invokeMethod(cameraWorker_, "stopCapture", Qt::DirectConnection);
    }
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
    lastCameraHoverFrame_ = frame.image;
    const cv::Mat previewMat = makeCameraPreviewMat(frame.image);
    previewMat.convertTo(lastRawCameraFrame_, CV_32F);
    cameraPreviewScaleX_ = frame.image.cols > 0
        ? static_cast<double>(lastRawCameraFrame_.cols) / static_cast<double>(frame.image.cols)
        : 1.0;
    cameraPreviewScaleY_ = frame.image.rows > 0
        ? static_cast<double>(lastRawCameraFrame_.rows) / static_cast<double>(frame.image.rows)
        : 1.0;

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
    lastCameraHoverFrame_ = frame.image;
    const qint64 nowMs = frame.timestampMs > 0
        ? frame.timestampMs
        : QDateTime::currentMSecsSinceEpoch();
    constexpr qint64 kCtPreviewIntervalMs = 50;
    if (cameraPreviewFrameCount_ > 1
        && lastCameraPreviewRenderMs_ > 0
        && nowMs - lastCameraPreviewRenderMs_ < kCtPreviewIntervalMs)
    {
        return;
    }
    lastCameraPreviewRenderMs_ = nowMs;
    const cv::Mat previewMat = makeCameraPreviewMat(frame.image);
    previewMat.convertTo(lastRawCameraFrame_, CV_32F);
    cameraPreviewScaleX_ = frame.image.cols > 0
        ? static_cast<double>(lastRawCameraFrame_.cols) / static_cast<double>(frame.image.cols)
        : 1.0;
    cameraPreviewScaleY_ = frame.image.rows > 0
        ? static_cast<double>(lastRawCameraFrame_.rows) / static_cast<double>(frame.image.rows)
        : 1.0;

    const PreviewWindow estimatedWindow = PreviewSliceRenderer::estimateRobustWindow(lastRawCameraFrame_);
    const bool windowOutsideData = estimatedWindow.isValid()
        && currentCameraPreviewWindow_.isValid()
        && (currentCameraPreviewWindow_.upper <= estimatedWindow.lower
            || currentCameraPreviewWindow_.lower >= estimatedWindow.upper);
    if (!currentCameraPreviewWindow_.isValid() || cameraPreviewFrameCount_ == 1 || windowOutsideData)
    {
        currentCameraPreviewWindow_ = estimatedWindow;
    }
    if (cameraHistogramWidget_)
    {
        const QSignalBlocker blocker(cameraHistogramWidget_);
        cameraHistogramWidget_->updateData(lastRawCameraFrame_);
        cameraHistogramWidget_->setWindow(currentCameraPreviewWindow_, false);
        currentCameraPreviewWindow_ = cameraHistogramWidget_->window();
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
    if (previewWindowProducesClippedImage(lastRawCameraFrame_, currentCameraPreviewWindow_))
    {
        const PreviewWindow recoveredWindow = PreviewSliceRenderer::estimateRobustWindow(lastRawCameraFrame_);
        if (recoveredWindow.isValid())
        {
            currentCameraPreviewWindow_ = recoveredWindow;
            if (cameraHistogramWidget_)
            {
                const QSignalBlocker blocker(cameraHistogramWidget_);
                cameraHistogramWidget_->updateData(lastRawCameraFrame_);
                cameraHistogramWidget_->setWindow(currentCameraPreviewWindow_, false);
                currentCameraPreviewWindow_ = cameraHistogramWidget_->window();
            }
        }
    }
    if (!currentCameraPreviewWindow_.isValid())
    {
        appendLog(QStringLiteral("Kinetix preview render skipped: invalid display window."));
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
        if (cameraPreviewFrameCount_ == 1)
        {
            appendLog(QStringLiteral("Kinetix preview rendered first frame: %1x%2.")
                .arg(image.width())
                .arg(image.height()));
        }
    }
    else
    {
        appendLog(QStringLiteral("Kinetix preview render failed: QImage is null."));
    }
}

void MainWindow::updateCameraPixelHover(int previewX, int previewY)
{
    if (!cameraPixelLabel_)
    {
        return;
    }
    const cv::Mat& source = !lastCameraHoverFrame_.empty() ? lastCameraHoverFrame_ : lastRawCameraFrame_;
    if (source.empty())
    {
        cameraPixelLabel_->setText(QStringLiteral("Camera x=- y=- value=-"));
        return;
    }
    const int sourceX = std::clamp(static_cast<int>(std::floor(static_cast<double>(previewX) / std::max(1e-9, cameraPreviewScaleX_))),
        0, source.cols - 1);
    const int sourceY = std::clamp(static_cast<int>(std::floor(static_cast<double>(previewY) / std::max(1e-9, cameraPreviewScaleY_))),
        0, source.rows - 1);
    cameraPixelLabel_->setText(QStringLiteral("Camera x=%1 y=%2 value=%3")
        .arg(sourceX)
        .arg(sourceY)
        .arg(pixelValueText(source, sourceX, sourceY)));
}

void MainWindow::updateReconPixelHover(int x, int y)
{
    if (!reconPixelLabel_)
    {
        return;
    }
    if (lastRawReconPreviewSlice_.empty())
    {
        reconPixelLabel_->setText(QStringLiteral("Recon x=- y=- value=-"));
        return;
    }
    const int sourceX = std::clamp(x, 0, lastRawReconPreviewSlice_.cols - 1);
    const int sourceY = std::clamp(y, 0, lastRawReconPreviewSlice_.rows - 1);
    reconPixelLabel_->setText(QStringLiteral("Recon x=%1 y=%2 value=%3")
        .arg(sourceX)
        .arg(sourceY)
        .arg(pixelValueText(lastRawReconPreviewSlice_, sourceX, sourceY)));
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
    if (ctRecoveryPending_ && !ctRecoveryConfirmed_)
    {
        appendLog(QStringLiteral("CT acquisition blocked: previous interrupted session has not been confirmed idle. Wait for CT recovery or reconnect Kohzu."));
        return;
    }
    if (!cameraWorker_ || !ctWriter_)
    {
        appendLog(QStringLiteral("CT acquisition failed: camera or writer is unavailable."));
        return;
    }
    const int requestedModeIndex = ctModeTabs_ ? ctModeTabs_->currentIndex() : 0;
    const bool requestedFlyMode = requestedModeIndex == kCtTabFly;
    const bool requestedTimeSeriesMode = requestedModeIndex == kCtTabTimeSeries;
    if (requestedModeIndex == kCtTabTiming)
    {
        appendLog(QStringLiteral("CT acquisition is unavailable on the Timing page. Run calibration, then select Fly or 2D."));
        return;
    }
    if ((requestedFlyMode || requestedTimeSeriesMode)
        && !(timingCalibrationValid_ && timingCalibrationMatchesCurrentUi()))
    {
        appendLog(QStringLiteral("CT acquisition failed: Timing calibration is required for the current Exposure and ROI."));
        if (ctModeTabs_)
        {
            ctModeTabs_->setCurrentIndex(kCtTabTiming);
        }
        return;
    }
    if (!requestedTimeSeriesMode && !motorConnected_)
    {
        appendLog(QStringLiteral("CT acquisition failed: Kohzu is not connected."));
        return;
    }
    const QString savePath = ctSavePathEdit_->text().trimmed();
    if (savePath.isEmpty())
    {
        appendLog(QStringLiteral("CT acquisition failed: Save Path is empty."));
        return;
    }
    if (directoryHasAnyEntries(savePath))
    {
        appendLog(QStringLiteral("CT acquisition failed: Save Path must be empty or absent to avoid overwriting existing CT data: %1").arg(savePath));
        return;
    }
    QDir().mkpath(savePath);
    if (!requestedTimeSeriesMode)
    {
        QDir().mkpath(QDir(savePath).filePath(QStringLiteral("recon")));
    }
    if (!QFileInfo(savePath).isDir())
    {
        appendLog(QStringLiteral("CT acquisition failed: Save Path is unavailable."));
        return;
    }
    if (!requestedTimeSeriesMode && (!motorAxes_.contains(QStringLiteral("R")) || !motorAxes_.contains(QStringLiteral("Y"))))
    {
        appendLog(QStringLiteral("CT acquisition failed: R/Y axes are unavailable."));
        return;
    }
    updateCtFlyDerivedParams();

    // The derived Fly speed is constrained by the configured R-axis top
    // speed.  updateCtFlyDerivedParams() already exposes this as a red
    // validation warning, but the acquisition path must enforce it before
    // arming PVCAM or sending WTB/APS.  Otherwise short exposures or very
    // small projection counts can produce multi-megapulse/s tables that the
    // ARIES controller rejects with WTB 607.
    if (requestedFlyMode)
    {
        const auto rIt = motorAxes_.constFind(QStringLiteral("R"));
        const double configuredTopSpeed = rIt != motorAxes_.constEnd() && rIt->speedSpin
            ? rIt->speedSpin->value()
            : 0.0;
        if (rIt == motorAxes_.constEnd() || configuredTopSpeed <= 0.0)
        {
            appendLog(QStringLiteral("CT acquisition failed: R Top Speed/s is not configured."));
            return;
        }
        if (ctFlyDerivedSpeedDegPerSec_ > configuredTopSpeed * (1.0 + 1e-9))
        {
            appendLog(QStringLiteral("CT acquisition failed: derived Fly speed %1 deg/s exceeds R Top Speed/s %2 deg/s; WTB was not sent.")
                .arg(ctFlyDerivedSpeedDegPerSec_, 0, 'f', 6)
                .arg(configuredTopSpeed, 0, 'f', 6));
            return;
        }
    }

    const MotorAxisUi* rAxis = requestedTimeSeriesMode ? nullptr : &motorAxes_[QStringLiteral("R")];
    const MotorAxisUi* yAxis = requestedTimeSeriesMode ? nullptr : &motorAxes_[QStringLiteral("Y")];
    ctSettings_ = {};
    ctSettings_.mode = requestedTimeSeriesMode
        ? CtAcquisitionMode::TimeSeries2D
        : (requestedFlyMode
        ? CtAcquisitionMode::FlyScan
        : CtAcquisitionMode::StepAndShoot);
    ctSettings_.savePath = savePath;
    ctSettings_.beginningAngleDeg = ctBeginningAngleSpin_->value();
    ctSettings_.projectionCount = ctProjectionNumSpin_->value();
    ctSettings_.flatGap = requestedFlyMode ? 0 : ctStepFlatGapSpin_->value();
    ctSettings_.flatMoveDistanceUm = requestedFlyMode ? ctFlyFlatMoveSpin_->value() : ctStepFlatMoveSpin_->value();
    ctSettings_.flatCount = requestedFlyMode ? ctFlyFlatNumSpin_->value() : ctStepFlatNumSpin_->value();
    ctSettings_.flySpeedDegPerSec = std::max(1e-6, ctFlyDerivedSpeedDegPerSec_);
    ctSettings_.flyLeadAngleDeg = std::max(0.0, ctFlyDerivedLeadAngleDeg_);
    ctSettings_.flyFrameIntervalMs = ctFlyDerivedFrameIntervalMs_;
    ctSettings_.flyScanTimeS = ctFlyDerivedScanTimeS_;
    ctSettings_.timeSeriesRawFrameStride = ctTimeSeriesRawFrameStrideSpin_
        ? ctTimeSeriesRawFrameStrideSpin_->value()
        : 1;
    ctSettings_.timeSeriesFrameIntervalMs = ctTimeSeriesActualIntervalMs_;
    if (requestedFlyMode || requestedTimeSeriesMode)
    {
        ctSettings_.rawFramePeriodMs = timingCalibrationResult_.analysis.fittedPeriodMs;
        ctSettings_.rawFrameRateFps = timingCalibrationResult_.analysis.rawFrameRateFps;
        ctSettings_.rawFrameStride = requestedFlyMode ? ctFlyRawFrameStride_ : ctTimeSeriesRawFrameStride_;
        ctSettings_.timingSource = timingCalibrationResult_.timingSource;
    }
    ctSettings_.autoCor = !requestedTimeSeriesMode;
    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        ctSettings_.flatGap = 0;
    }
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        ctSettings_.flatGap = 0;
        ctSettings_.flatMoveDistanceUm = 0.0;
        ctSettings_.flatCount = 0;
    }
    ctSettings_.padding = paddingSpin_->value();
    ctSettings_.tomoPrefix = tomoPrefixEdit_->text().isEmpty() ? QStringLiteral("Tomo_") : tomoPrefixEdit_->text();
    ctSettings_.flatPrefix = flatPrefixEdit_->text().isEmpty() ? QStringLiteral("Flat_") : flatPrefixEdit_->text();
    ctSettings_.timeSeriesPrefix = QStringLiteral("Frame_");
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
    ctSettings_.camera.framePoolCapacity = kCtFramePoolCapacity;
    ctSettings_.camera.ringFrameCount = 192;
    ctSettings_.camera.baselineRawFramePeriodMs = ctSettings_.rawFramePeriodMs;
    ctSettings_.camera.saveRaw = false;
    ctSettings_.camera.previewOnly = false;
    ctSettings_.camera.enqueueFrames = false;
    ctSettings_.camera.outputDir = savePath;
    ctSettings_.camera.previewIntervalMs = 0;
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        ctSettings_.camera.frameCount = std::max(1, ctSettings_.projectionCount);
    }
    ctSettings_.camera.frameIntervalMs = 0.0;
    ctSettings_.camera.firstKeepDelayMs = ctSettings_.mode == CtAcquisitionMode::FlyScan
        ? ctFlyFirstKeepDelayMs_
        : 0.0;
    ctSettings_.camera.rawFrameStride = std::max(1, ctSettings_.rawFrameStride);
    ctSettings_.camera.waitForFrameGate = ctSettings_.mode == CtAcquisitionMode::FlyScan;
    ctSettings_.camera.allowFrameIntervalDecimation = false;
    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        if (ctSettings_.flySpeedDegPerSec <= 0.0 || ctSettings_.flyScanTimeS <= 0.0
            || ctSettings_.flyFrameIntervalMs <= 0.0 || ctSettings_.rawFrameStride <= 0)
        {
            appendLog(QStringLiteral("CT acquisition failed: FlyScan timing plan is invalid. Run Timing calibration again."));
            return;
        }
    }
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        if (ctSettings_.projectionCount <= 0)
        {
            appendLog(QStringLiteral("2D acquisition failed: Projection Num must be positive."));
            return;
        }
        if (ctSettings_.timeSeriesFrameIntervalMs <= 0.0 || ctSettings_.rawFrameStride <= 0)
        {
            appendLog(QStringLiteral("2D acquisition failed: integer-stride timing plan is invalid. Run Timing calibration again."));
            return;
        }
    }
    ctSettings_.reconConfig = configFromUi();
    if (ctSettings_.reconConfig.io.reconMethod == ReconMethod::RealTimeFBP)
    {
        ctSettings_.reconConfig.io.reconMethod = ReconMethod::OfflineFBP;
    }
    ctSettings_.reconConfig.geometry.corAuto = ctSettings_.mode != CtAcquisitionMode::TimeSeries2D;
    ctSettings_.reconConfig.geometry.corShift = 0.0f;
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
    if (rAxis && yAxis)
    {
        ctSettings_.rAxis = rAxis->axis;
        ctSettings_.yAxis = yAxis->axis;
        ctSettings_.rPulsePerDeg = rAxis->pulsePerUnit;
        ctSettings_.yPulsePerUm = yAxis->pulsePerUnit;

        const double startCheckAngle = ctSettings_.mode == CtAcquisitionMode::FlyScan
            ? ctSettings_.beginningAngleDeg - ctSettings_.flyLeadAngleDeg
            : ctSettings_.beginningAngleDeg;
        const double endAngle = ctSettings_.mode == CtAcquisitionMode::FlyScan
            ? ctSettings_.beginningAngleDeg + 180.0 + ctSettings_.flyLeadAngleDeg
            : ctSettings_.beginningAngleDeg + 180.0;
        if (startCheckAngle < rAxis->softMin || startCheckAngle > rAxis->softMax
            || endAngle < rAxis->softMin || endAngle > rAxis->softMax)
        {
            appendLog(QStringLiteral("CT acquisition failed: R range %1..%2 %3 exceeds soft limit [%4, %5].")
                .arg(startCheckAngle, 0, 'f', 4)
                .arg(endAngle, 0, 'f', 4)
                .arg(rAxis->unit)
                .arg(rAxis->softMin, 0, 'f', 4)
                .arg(rAxis->softMax, 0, 'f', 4));
            return;
        }
        const double yFlatTarget = yAxis->currentValue + ctSettings_.flatMoveDistanceUm;
        if (ctSettings_.flatCount > 0 && (yFlatTarget < yAxis->softMin || yFlatTarget > yAxis->softMax))
        {
            appendLog(QStringLiteral("CT acquisition failed: flat Y target %1 %2 exceeds soft limit [%3, %4].")
                .arg(yFlatTarget, 0, 'f', 4)
                .arg(yAxis->unit)
                .arg(yAxis->softMin, 0, 'f', 4)
                .arg(yAxis->softMax, 0, 'f', 4));
            return;
        }
    }

    ctRestoreLiveAfterFinish_ = cameraLiveRunning_;
    if (cameraLiveRunning_)
    {
        appendLog(QStringLiteral("Kinetix Live is running; closing it before CT acquisition and will restore it afterwards."));
        stopCameraLivePreview(false);
    }

    ctManifestPath_ = QDir(savePath).filePath(QStringLiteral("scan_manifest.json"));
    ctFlatGroups_.clear();
    ctFlyFrames_.clear();
    ctCor180Frame_ = {};
    ctTimeSeriesFrames_.clear();
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
    ctPendingFrameBatchRemaining_ = 0;
    ctProjectionNeedsMove_ = false;
    ctCorReferencePending_ = false;
    ctCorReferenceCaptured_ = false;
    ctFlyScanRequested_ = false;
    ctFlyScanCapturing_ = false;
    ctFlyScanMoveDone_ = false;
    ctFlyRestartingPreparedAfterScan_ = false;
    ctTimeSeriesCapturing_ = false;
    ctContinuousStartPending_ = false;
    ctContinuousCaptureActive_ = false;
    ctContinuousIntervalStrategy_.clear();
    ctContinuousIntervalFallbackReason_.clear();
    ctContinuousTimingStats_ = {};
    ctContinuousTimingDriftLevel_ = 0;
    ctPvcamRingEstimatedUnread_ = 0;
    ctPvcamRingCapacity_ = 0;
    ctPvcamRingPeakEstimatedUnread_ = 0;
    ctPvcamRingLastUiUpdateMs_ = 0;
    if (ctPvcamRingLabel_)
    {
        ctPvcamRingLabel_->setText(QStringLiteral("Inactive"));
    }
    ctPendingFinishErrorMessage_.clear();
    ctFlyFlatStage_ = 0;
    ctFlyContinuousFramesReceived_ = 0;
    ctFlyRawFramePlan_ = 0;
    ctFlyDerivedScanTimeS_ = ctSettings_.flyScanTimeS;
    ctFlyDerivedFrameIntervalMs_ = ctSettings_.flyFrameIntervalMs;
    ctFlyScanStartedMs_ = 0;
    ctFlyScanFinishedMs_ = 0;
    ctPendingMotorAction_.clear();
    ctAcquisitionStartedMs_ = QDateTime::currentMSecsSinceEpoch();
    ctFirstFrameRequestedMs_ = 0;
    ctLastFrameEnqueuedMs_ = 0;
    ctWriterFinishRequestedMs_ = 0;
    ctPreparedStartedMs_ = 0;
    ctPreparedReadyMs_ = 0;
    ctCurrentFrameBatchStartedMs_ = 0;
    ctCurrentMotorMoveStartedMs_ = 0;
    ctPreparedSetupMs_ = 0;
    ctMotorMoveMs_ = 0;
    ctRMoveMs_ = 0;
    ctYMoveMs_ = 0;
    ctFrameCaptureMs_ = 0;
    ctFrameEnqueueMs_ = 0;
    ctWriterDrainMs_ = 0;
    ctCompletedMs_ = 0;
    ctMotorMoveCount_ = 0;
    ctRMoveCount_ = 0;
    ctYMoveCount_ = 0;
    ctFrameBatchCount_ = 0;
    ctFrameSavedCount_ = 0;
    ctContinuousRawFrames_ = 0;
    ctContinuousDiscardedFrames_ = 0;
    ctContinuousDeviceDroppedFrames_ = 0;
    ctFramePoolInUse_ = 0;
    ctFramePoolPeak_ = 0;
    ctMotorSpeedTablesReady_ = false;
    ctMotorSpeedSetupStage_ = 0;
    ctIdlePollingPausedForCt_ = false;
    ctEstimatedTotalFrames_ = estimateCtTotalFrames();
    ctEstimatedTotalSeconds_ = estimateCtAcquisitionSeconds(ctEstimatedTotalFrames_);
    ctRunning_ = true;
    writeCtRecoveryState(QStringLiteral("RUNNING"), QStringLiteral("%1 acquisition started.")
        .arg(ctSettings_.mode == CtAcquisitionMode::FlyScan ? QStringLiteral("FlyScan") : QStringLiteral("StepAndShoot")));

    ctStartButton_->setEnabled(false);
    ctStopButton_->setEnabled(true);
    ctStatusLabel_->setText(QStringLiteral("Running"));
    ctStatusLabel_->setStyleSheet(QString());
    updateContinuousTimingUi();
    if (ctProgressTimer_)
    {
        ctProgressTimer_->start();
    }
    updateCtProgressUi();
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
    if (ctModeTabs_) { ctModeTabs_->setEnabled(false); }
    ctSavePathEdit_->setEnabled(false);
    ctProjectionNumSpin_->setEnabled(false);
    ctStepFlatGapSpin_->setEnabled(false);
    ctStepFlatMoveSpin_->setEnabled(false);
    ctStepFlatNumSpin_->setEnabled(false);
    ctFlyFlatMoveSpin_->setEnabled(false);
    ctFlyFlatNumSpin_->setEnabled(false);
    ctTiffThreadsSpin_->setEnabled(false);
    if (ctFlyAllowedBlurSpin_) { ctFlyAllowedBlurSpin_->setEnabled(false); }
    if (ctTimeSeriesRawFrameStrideSpin_) { ctTimeSeriesRawFrameStrideSpin_->setEnabled(false); }
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;

    // A continuous frame stays leased while it waits in the TIFF queue and
    // while a writer thread is writing it. Keep one slot free for the camera
    // thread itself, otherwise a full writer queue can exhaust the frame pool
    // before the next ring frame has been copied and unlocked.
    const int requestedWriterThreads = std::max(1, ctSettings_.tiffThreads);
    ctWriterThreads_ = std::min(requestedWriterThreads, kAcquisitionWriterThreadLimit);
    if (ctWriterThreads_ != requestedWriterThreads)
    {
        appendLog(QStringLiteral("Acquisition limits TIFF writers to %1 threads (requested %2).")
            .arg(ctWriterThreads_)
            .arg(requestedWriterThreads));
    }
    const int maxQueued = std::max(1, kCtFramePoolCapacity - ctWriterThreads_ - 2);
    ctWriterQueueCapacity_ = maxQueued;
    ctWriterFrameBytes_ = static_cast<qint64>(std::max(1, ctSettings_.camera.roiWidth))
        * static_cast<qint64>(std::max(1, ctSettings_.camera.roiHeight)) * static_cast<qint64>(2);
    ctWriterStartedMs_ = QDateTime::currentMSecsSinceEpoch();
    const double capacityGiB = (static_cast<double>(ctWriterQueueCapacity_) * static_cast<double>(ctWriterFrameBytes_))
        / (1024.0 * 1024.0 * 1024.0);
    if (ctWriterLabel_)
    {
        ctWriterLabel_->setText(QStringLiteral("%1 threads, 0.0 MiB/s, saved 0").arg(ctWriterThreads_));
    }
    if (ctBufferLabel_)
    {
        ctBufferLabel_->setText(QStringLiteral("0 / %1 frame pool slots, 0.0 / %2 GiB")
            .arg(kCtFramePoolCapacity)
            .arg(capacityGiB, 0, 'f', 1));
    }
    // The first PVCAM frame can arrive as soon as continuous capture starts.
    // Start the writer before arming the camera so the direct router never
    // observes a writer that is not yet accepting frames.
    QMetaObject::invokeMethod(ctWriter_, [writer = ctWriter_, threads = ctWriterThreads_, maxQueued]() {
        writer->configure(threads, maxQueued);
        writer->start();
    }, Qt::BlockingQueuedConnection);

    if (ctContinuousRouter_ && cameraWorker_)
    {
        if (ctSettings_.mode == CtAcquisitionMode::FlyScan
            || ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
        {
            ContinuousFrameRouteSettings routeSettings;
            routeSettings.savePath = ctSettings_.savePath;
            routeSettings.tomoPrefix = ctSettings_.tomoPrefix;
            routeSettings.timeSeriesPrefix = ctSettings_.timeSeriesPrefix;
            routeSettings.alignment180Path = QDir(ctSettings_.savePath).filePath(ctSettings_.reconConfig.io.filePattern180);
            routeSettings.extension = ctSettings_.extension;
            routeSettings.padding = ctSettings_.padding;
            routeSettings.projectionCount = ctSettings_.projectionCount;
            routeSettings.flyScan = ctSettings_.mode == CtAcquisitionMode::FlyScan;
            routeSettings.autoCor = ctSettings_.autoCor;
            routeSettings.previewEnabled = qEnvironmentVariableIntValue("CT_SMOKE_DISABLE_CONTINUOUS_PREVIEW") != 1;
            routeSettings.previewIntervalMs = 50;
            ctContinuousRouter_->configure(routeSettings, ctWriter_);
            ctContinuousRouter_->start();
            // The actual base is captured immediately before the continuous
            // camera starts. Fly may save start flats after this router setup.
            ctContinuousFrameBaseCount_ = 0;
            cameraWorker_->setContinuousFrameSink(ctContinuousRouter_);
            appendLog(routeSettings.previewEnabled
                ? QStringLiteral("Continuous frame router enabled: direct writer path, latest preview every 50 ms.")
                : QStringLiteral("Continuous frame router enabled: direct writer path, preview disabled for smoke diagnosis."));
        }
        else
        {
            cameraWorker_->setContinuousFrameSink(nullptr);
            ctContinuousRouter_->stopPreview();
        }
    }

    const QString modeLabel = ctSettings_.mode == CtAcquisitionMode::FlyScan
        ? QStringLiteral("FlyScan")
        : (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D ? QStringLiteral("TimeSeries2D") : QStringLiteral("StepAndShoot"));
    appendLog(QStringLiteral("CT acquisition started: mode=%1, frames=%2, begin=%3 deg, step=%4 deg, flats gap=%5 num=%6, auto COR=%7.")
        .arg(modeLabel)
        .arg(ctSettings_.projectionCount)
        .arg(ctSettings_.beginningAngleDeg, 0, 'f', 4)
        .arg(180.0 / std::max(1, ctSettings_.projectionCount), 0, 'f', 6)
        .arg(ctSettings_.flatGap)
        .arg(ctSettings_.flatCount)
        .arg(ctSettings_.autoCor ? QStringLiteral("on") : QStringLiteral("off")));
    appendLog(QStringLiteral("CT frame pool: capacity=%1 slots, frame=%2 MiB, total=%3 GiB; writer queue capacity=%4 frames.")
        .arg(kCtFramePoolCapacity)
        .arg(static_cast<double>(ctWriterFrameBytes_) / (1024.0 * 1024.0), 0, 'f', 2)
        .arg(static_cast<double>(kCtFramePoolCapacity) * static_cast<double>(ctWriterFrameBytes_)
            / (1024.0 * 1024.0 * 1024.0), 0, 'f', 1)
        .arg(ctWriterQueueCapacity_));
    appendLog(QStringLiteral("CT estimate: frames=%1, total=%2.")
        .arg(ctEstimatedTotalFrames_)
        .arg(formatDurationSeconds(ctEstimatedTotalSeconds_)));
    writeCtManifest(false, QStringLiteral("CT acquisition running."));
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        startCtTimeSeriesCapture();
    }
    else
    {
        startCtMotorSpeedTableSetup();
    }
    const int smokeStopAfterMs = qEnvironmentVariableIntValue("CT_SMOKE_STOP_AFTER_MS");
    if (qEnvironmentVariableIntValue("CT_SMOKE_AUTOSTART") == 1 && smokeStopAfterMs > 0)
    {
        appendLog(QStringLiteral("CT smoke will request Stop after %1 ms.").arg(smokeStopAfterMs));
        QTimer::singleShot(smokeStopAfterMs, this, [this]() {
            if (ctRunning_)
            {
                appendLog(QStringLiteral("CT smoke requesting Stop now."));
                stopCtAcquisition();
            }
        });
    }
}

void MainWindow::stopCtAcquisition()
{
    if (!ctRunning_)
    {
        return;
    }
    ctStopRequested_ = true;
    writeCtRecoveryState(QStringLiteral("STOPPING"), QStringLiteral("User requested CT stop; hardware shutdown in progress."));
    ctRestoreLiveAfterFinish_ = false;
    ++liveRestoreGeneration_;
    appendLog(QStringLiteral("Stopping CT acquisition."));
    if (ctReplayRunning_)
    {
        if (ctReplayWorker_)
        {
            QMetaObject::invokeMethod(ctReplayWorker_, "stopReplay", Qt::QueuedConnection);
        }
        if (ctWriter_)
        {
            QMetaObject::invokeMethod(ctWriter_, "stop", Qt::QueuedConnection);
        }
        ctReplayJobQueue_.clear();
        ctReplayTargetQueue_.clear();
        ctReplayOutputQueue_.clear();
        finishCtAcquisition(false, QStringLiteral("Internal CT replay stopped by user."));
        return;
    }
    if (cameraWorker_)
    {
        QMetaObject::invokeMethod(cameraWorker_, "stopCapture", Qt::DirectConnection);
        QMetaObject::invokeMethod(cameraWorker_, "endPreparedCapture", Qt::QueuedConnection);
    }
    if (kohzuController_ && ctSettings_.mode != CtAcquisitionMode::TimeSeries2D)
    {
        ctHardwareStopPending_ = true;
        QMetaObject::invokeMethod(kohzuController_, "stopAllAxes", Qt::QueuedConnection);
    }
    if (ctWriter_)
    {
        if (ctWriterFinishRequestedMs_ <= 0)
        {
            ctWriterFinishRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
        }
        QMetaObject::invokeMethod(ctWriter_, "finish", Qt::QueuedConnection);
    }
}

void MainWindow::maybeStartCtSmokeAutostart()
{
    writeStartupTrace(QStringLiteral("maybeStartCtSmokeAutostart enter"));
    if (qEnvironmentVariableIntValue("CT_SMOKE_AUTOSTART") != 1)
    {
        writeStartupTrace(QStringLiteral("maybeStartCtSmokeAutostart disabled"));
        return;
    }
    if (ctRunning_ || controller_.isBusy() || ctSmokeAcquisitionStarted_ || ctSmokeWaitingForTiming_)
    {
        writeStartupTrace(QStringLiteral("maybeStartCtSmokeAutostart deferred: busy"));
        return;
    }

    const QString smokeMode = qEnvironmentVariable("CT_SMOKE_MODE", QStringLiteral("step")).trimmed().toLower();
    const bool stepMode = smokeMode == QStringLiteral("step");
    const bool timingMode = smokeMode == QStringLiteral("timing");
    const bool flyMode = smokeMode == QStringLiteral("fly");
    const bool timeSeriesMode = smokeMode == QStringLiteral("2d") || smokeMode == QStringLiteral("timeseries2d");
    if (!stepMode && !timingMode && !flyMode && !timeSeriesMode)
    {
        appendLog(QStringLiteral("CT smoke autostart failed: unsupported CT_SMOKE_MODE '%1'. Use step, timing, fly, or 2d.")
            .arg(smokeMode));
        return;
    }

    const bool needsMotor = stepMode || flyMode;
    const bool axesReady = motorAxes_.contains(QStringLiteral("R"))
        && motorAxes_.contains(QStringLiteral("Y"))
        && motorAxes_[QStringLiteral("R")].hasPosition
        && motorAxes_[QStringLiteral("Y")].hasPosition;
    const bool hardwareReady = cameraWorker_ && (!needsMotor || (motorConnected_ && axesReady));
    if (!hardwareReady && ctSmokeAutostartAttempts_ < 60)
    {
        ++ctSmokeAutostartAttempts_;
        if ((ctSmokeAutostartAttempts_ % 5) == 1)
        {
            appendLog(QStringLiteral("CT smoke autostart waiting: mode=%1, cameraWorker=%2, motorConnected=%3, axesReady=%4.")
                .arg(smokeMode)
                .arg(cameraWorker_ ? QStringLiteral("true") : QStringLiteral("false"))
                .arg(motorConnected_ ? QStringLiteral("true") : QStringLiteral("false"))
                .arg(axesReady ? QStringLiteral("true") : QStringLiteral("false")));
        }
        QTimer::singleShot(1000, this, &MainWindow::maybeStartCtSmokeAutostart);
        return;
    }
    if (!hardwareReady)
    {
        appendLog(QStringLiteral("CT smoke autostart failed: hardware did not become ready."));
        return;
    }

    if (ctSmokeParametersApplied_)
    {
        return;
    }
    ctSmokeParametersApplied_ = true;

    const QString defaultSmokeRoot = QDir(QStringLiteral("F:/")).filePath(
        QStringLiteral("ct_smoke_%1_%2")
            .arg(smokeMode)
            .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"))));
    const QString savePath = qEnvironmentVariable("CT_SMOKE_SAVE_PATH", defaultSmokeRoot);
    writeStartupTrace(QStringLiteral("CT smoke save path len=%1 native=%2")
        .arg(savePath.size())
        .arg(QDir::fromNativeSeparators(savePath)));

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
    ctSmokeTargetTab_ = stepMode ? kCtTabStep
        : (timingMode ? kCtTabTiming : (flyMode ? kCtTabFly : kCtTabTimeSeries));
    if (ctModeTabs_) { ctModeTabs_->setCurrentIndex(stepMode ? kCtTabStep : kCtTabTiming); }

    ctSavePathEdit_->setText(savePath);
    ctBeginningAngleSpin_->setValue(qEnvironmentVariable("CT_SMOKE_BEGIN_ANGLE").isEmpty()
            ? -180.0
            : qEnvironmentVariable("CT_SMOKE_BEGIN_ANGLE").toDouble());
    ctProjectionNumSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_PROJECTIONS") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_PROJECTIONS")
            : 5);
    ctStepFlatGapSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_FLAT_GAP") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_FLAT_GAP")
            : 2);
    ctStepFlatMoveSpin_->setValue(qEnvironmentVariable("CT_SMOKE_FLAT_MOVE_UM").isEmpty()
            ? (flyMode ? 10000.0 : 10.0)
            : qEnvironmentVariable("CT_SMOKE_FLAT_MOVE_UM").toDouble());
    ctStepFlatNumSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_FLAT_NUM") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_FLAT_NUM")
            : (flyMode ? 2 : 1));
    ctTiffThreadsSpin_->setValue(qEnvironmentVariableIntValue("CT_SMOKE_TIFF_THREADS") > 0
            ? qEnvironmentVariableIntValue("CT_SMOKE_TIFF_THREADS")
            : 4);
    if (ctFlyAllowedBlurSpin_ && !qEnvironmentVariable("CT_SMOKE_FLY_BLUR_PX").isEmpty())
    {
        ctFlyAllowedBlurSpin_->setValue(qEnvironmentVariable("CT_SMOKE_FLY_BLUR_PX").toDouble());
    }
    if (ctTimeSeriesRawFrameStrideSpin_ && !qEnvironmentVariable("CT_SMOKE_2D_RAW_STRIDE").isEmpty())
    {
        ctTimeSeriesRawFrameStrideSpin_->setValue(std::max(1, qEnvironmentVariableIntValue("CT_SMOKE_2D_RAW_STRIDE")));
    }
    updateCtFlyDerivedParams();
    updateCtTimeSeriesDerivedParams();

    correctionModeCombo_->setCurrentText(toString(CorrectionMode::Flat));
    reconMethodCombo_->setCurrentText(toString(ReconMethod::OfflineFBP));
    batchSpin_->setValue(4);
    zChunkSpin_->setValue(16);
    tiffThreadsSpin_->setValue(4);

    appendLog(QStringLiteral("CT smoke autostart: mode=%1, path=%2, ROI=%3x%4, exposure=%5 ms, frames=%6, flatGap=%7, flatMove=%8 um.")
        .arg(smokeMode)
        .arg(savePath)
        .arg(cameraRoiWidthSpin_->value())
        .arg(cameraRoiHeightSpin_->value())
        .arg(cameraExposureSpin_->value(), 0, 'f', 4)
        .arg(ctProjectionNumSpin_->value())
        .arg(ctStepFlatGapSpin_->value())
        .arg(ctStepFlatMoveSpin_->value(), 0, 'f', 4));
    if (stepMode)
    {
        ctSmokeAcquisitionStarted_ = true;
        writeStartupTrace(QStringLiteral("CT smoke starting Step acquisition"));
        startCtAcquisition();
        return;
    }

    ctSmokeWaitingForTiming_ = true;
    timingRestoreLiveAfterFinish_ = false;
    writeStartupTrace(QStringLiteral("CT smoke starting Timing calibration"));
    startTimingCalibration();
}

void MainWindow::maybeStartCtReplayAutostart()
{
    if (qEnvironmentVariableIntValue("CT_REPLAY_AUTOSTART") != 1)
    {
        return;
    }
    if (ctRunning_ || controller_.isBusy())
    {
        QTimer::singleShot(1000, this, &MainWindow::maybeStartCtReplayAutostart);
        return;
    }
    if (ctReplayJobQueue_.isEmpty())
    {
        const QString repoRoot = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("../.."));
        const QString singleJob = qEnvironmentVariable("CT_REPLAY_JOB");
        if (!singleJob.isEmpty())
        {
            ctReplayJobQueue_.append(QDir::fromNativeSeparators(singleJob));
            ctReplayTargetQueue_.append(QDir::fromNativeSeparators(qEnvironmentVariable("CT_REPLAY_TARGET", QStringLiteral("F:/Stack1"))));
            ctReplayOutputQueue_.append(QDir::fromNativeSeparators(qEnvironmentVariable("CT_REPLAY_OUTPUT", QStringLiteral("F:/FenMu"))));
        }
        else
        {
            ctReplayJobQueue_ = {
                QDir(repoRoot).filePath(QStringLiteral("build/vs2022-release/recon_queue/done/000010_32_.json")),
                QDir(repoRoot).filePath(QStringLiteral("build/vs2022-release/recon_queue/done/000010_2-1.json")),
                QDir(repoRoot).filePath(QStringLiteral("build/vs2022-release/recon_queue/done/000010_test.json")),
            };
            ctReplayTargetQueue_ = { QStringLiteral("F:/Stack1"), QStringLiteral("F:/Stack2"), QStringLiteral("F:/Stack3") };
            ctReplayOutputQueue_ = { QStringLiteral("F:/FenMu"), QStringLiteral("F:/FenMu2"), QStringLiteral("F:/FenMu3") };
        }
    }
    startInternalCtReplay();
}

void MainWindow::startInternalCtReplay()
{
    if (ctRunning_)
    {
        appendLog(QStringLiteral("Internal CT replay skipped: CT acquisition is already running."));
        return;
    }
    if (!ctWriter_)
    {
        appendLog(QStringLiteral("Internal CT replay failed: CT writer is unavailable."));
        return;
    }

    QString sourceJob;
    QString targetProjectionDir;
    QString targetOutputDir;
    for (;;)
    {
        if (ctReplayJobQueue_.isEmpty())
        {
            appendLog(QStringLiteral("Internal CT replay completed all queued datasets."));
            return;
        }
        if (ctReplayTargetQueue_.isEmpty() || ctReplayOutputQueue_.isEmpty())
        {
            appendLog(QStringLiteral("Internal CT replay failed: replay job/target/output queues are out of sync."));
            ctReplayJobQueue_.clear();
            ctReplayTargetQueue_.clear();
            ctReplayOutputQueue_.clear();
            return;
        }

        sourceJob = ctReplayJobQueue_.takeFirst();
        targetProjectionDir = QDir::fromNativeSeparators(ctReplayTargetQueue_.takeFirst());
        targetOutputDir = QDir::fromNativeSeparators(ctReplayOutputQueue_.takeFirst());

        const bool projectionDirHasFiles = !QDir(targetProjectionDir)
            .entryList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System)
            .isEmpty();
        if (projectionDirHasFiles)
        {
            QJsonObject existingManifest;
            QString existingError;
            if (readJsonFile(manifestPathForProjectionDir(targetProjectionDir), &existingManifest, &existingError)
                && existingManifest.value(QStringLiteral("COMPLETED")).toBool(false))
            {
                appendLog(QStringLiteral("Internal CT replay skipping completed target projection dir: %1").arg(targetProjectionDir));
                continue;
            }
            appendLog(QStringLiteral("Internal CT replay failed: target projection dir is not empty or completed: %1").arg(targetProjectionDir));
            return;
        }
        if (!QDir(targetOutputDir).entryList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System).isEmpty())
        {
            appendLog(QStringLiteral("Internal CT replay failed: target output dir is not empty: %1").arg(targetOutputDir));
            return;
        }
        break;
    }

    QJsonObject sourceJobObj;
    QString error;
    if (!readJsonFile(sourceJob, &sourceJobObj, &error))
    {
        appendLog(QStringLiteral("Internal CT replay failed: %1").arg(error));
        return;
    }
    const QString sourceProjectionDir = projectionDirFromJob(sourceJobObj);
    QJsonObject manifest;
    if (!readJsonFile(manifestPathForProjectionDir(sourceProjectionDir), &manifest, &error))
    {
        appendLog(QStringLiteral("Internal CT replay failed: %1").arg(error));
        return;
    }
    if (!manifest.value(QStringLiteral("COMPLETED")).toBool(false))
    {
        appendLog(QStringLiteral("Internal CT replay failed: source manifest is not completed."));
        return;
    }

    const QJsonObject acq = manifest.value(QStringLiteral("ACQUISITION")).toObject();
    const QJsonObject io = manifest.value(QStringLiteral("IO")).toObject();
    QVector<CtFlatGroup> replayFlatGroups;
    for (const QJsonValue& value : manifest.value(QStringLiteral("FLAT_GROUPS")).toArray())
    {
        const QJsonObject obj = value.toObject();
        CtFlatGroup group;
        group.label = obj.value(QStringLiteral("LABEL")).toString();
        group.afterProjectionCount = obj.value(QStringLiteral("AFTER_PROJECTION_COUNT")).toInt();
        group.firstFlatIndex = obj.value(QStringLiteral("FIRST_FLAT_INDEX")).toInt();
        group.flatCount = obj.value(QStringLiteral("FLAT_COUNT")).toInt();
        group.yMoveUm = obj.value(QStringLiteral("Y_MOVE_UM")).toDouble();
        if (group.flatCount > 0)
        {
            replayFlatGroups.append(group);
        }
    }

    AppConfig reconConfig = configFromUi();
    applyJobObjectToConfig(sourceJobObj, reconConfig);
    if (reconConfig.io.reconMethod == ReconMethod::RealTimeFBP)
    {
        reconConfig.io.reconMethod = ReconMethod::OfflineFBP;
    }
    reconConfig.io.watchDir = targetProjectionDir;
    reconConfig.io.saveDir = targetOutputDir;
    reconConfig.io.tomoPrefix = io.value(QStringLiteral("TOMO_PREFIX")).toString(reconConfig.io.tomoPrefix);
    reconConfig.io.flatPrefix = io.value(QStringLiteral("FLAT_PREFIX")).toString(reconConfig.io.flatPrefix);
    reconConfig.io.darkPrefix = io.value(QStringLiteral("DARK_PREFIX")).toString(reconConfig.io.darkPrefix);
    reconConfig.io.inputPadding = io.value(QStringLiteral("INPUT_PADDING")).toInt(reconConfig.io.inputPadding);
    reconConfig.io.inputExt = io.value(QStringLiteral("INPUT_EXT")).toString(reconConfig.io.inputExt);
    reconConfig.io.filePattern0 = io.value(QStringLiteral("FILE_PATTERN_0")).toString(reconConfig.io.filePattern0);
    reconConfig.io.filePattern180 = io.value(QStringLiteral("FILE_PATTERN_180")).toString(reconConfig.io.filePattern180);
    reconConfig.geometry.totalAngles = acq.value(QStringLiteral("PROJECTION_COUNT")).toInt(reconConfig.geometry.totalAngles);
    reconConfig.geometry.volW = acq.value(QStringLiteral("IMAGE_WIDTH")).toInt(reconConfig.geometry.volW);
    reconConfig.geometry.volH = acq.value(QStringLiteral("IMAGE_HEIGHT")).toInt(reconConfig.geometry.volH);
    reconConfig.geometry.angleOffsetDeg = static_cast<float>(acq.value(QStringLiteral("BEGINNING_ANGLE_DEG")).toDouble(reconConfig.geometry.angleOffsetDeg));
    reconConfig.geometry.corAuto = acq.value(QStringLiteral("AUTO_COR")).toBool(false);
    reconConfig.geometry.corShift = 0.0f;
    reconConfig.preprocess.flatGap = acq.value(QStringLiteral("FLAT_GAP")).toInt(reconConfig.preprocess.flatGap);
    reconConfig.preprocess.flatNum = acq.value(QStringLiteral("FLAT_NUM")).toInt(reconConfig.preprocess.flatNum);
    reconConfig.preprocess.flatGroups.clear();
    for (const CtFlatGroup& group : replayFlatGroups)
    {
        FlatGroupConfig cfgGroup;
        cfgGroup.label = group.label;
        cfgGroup.afterProjectionCount = group.afterProjectionCount;
        cfgGroup.firstFlatIndex = group.firstFlatIndex;
        cfgGroup.flatCount = group.flatCount;
        reconConfig.preprocess.flatGroups.append(cfgGroup);
    }
    if (qEnvironmentVariableIntValue("CT_REPLAY_FORCE_STRIPE") == 1)
    {
        reconConfig.stripe.enable = false;
        reconConfig.stripe.method = QStringLiteral("FW");
        if (reconConfig.stripe.waveletName.isEmpty())
        {
            reconConfig.stripe.waveletName = QStringLiteral("db5");
        }
        appendLog(QStringLiteral("Internal CT replay ignored CT_REPLAY_FORCE_STRIPE because FW stripe removal is temporarily disabled."));
    }

    ctSettings_ = {};
    ctSettings_.savePath = targetProjectionDir;
    ctSettings_.beginningAngleDeg = acq.value(QStringLiteral("BEGINNING_ANGLE_DEG")).toDouble();
    ctSettings_.projectionCount = reconConfig.geometry.totalAngles;
    ctSettings_.flatGap = reconConfig.preprocess.flatGap;
    ctSettings_.flatMoveDistanceUm = acq.value(QStringLiteral("FLAT_MOVE_DISTANCE_UM")).toDouble();
    ctSettings_.flatCount = reconConfig.preprocess.flatNum;
    ctSettings_.autoCor = reconConfig.geometry.corAuto;
    ctSettings_.padding = reconConfig.io.inputPadding;
    ctSettings_.tomoPrefix = reconConfig.io.tomoPrefix;
    ctSettings_.flatPrefix = reconConfig.io.flatPrefix;
    ctSettings_.extension = reconConfig.io.inputExt;
    ctSettings_.tiffThreads = qEnvironmentVariableIntValue("CT_REPLAY_TIFF_THREADS") > 0
        ? qEnvironmentVariableIntValue("CT_REPLAY_TIFF_THREADS")
        : ctTiffThreadsSpin_->value();
    ctSettings_.camera.roiWidth = reconConfig.geometry.volW;
    ctSettings_.camera.roiHeight = reconConfig.geometry.volH;
    ctSettings_.camera.roiX = manifest.value(QStringLiteral("CAMERA")).toObject().value(QStringLiteral("ROI_X")).toInt();
    ctSettings_.camera.roiY = manifest.value(QStringLiteral("CAMERA")).toObject().value(QStringLiteral("ROI_Y")).toInt();
    ctSettings_.reconConfig = reconConfig;

    ctManifestPath_ = QDir(targetProjectionDir).filePath(QStringLiteral("scan_manifest.json"));
    ctFlatGroups_ = replayFlatGroups;
    ctFlyFrames_.clear();
    ctCor180Frame_ = {};
    ctTimeSeriesFrames_.clear();
    ctProjectionIndex_ = 0;
    ctFlatIndex_ = 0;
    ctFlatFramesRemaining_ = 0;
    ctFlatAfterProjectionCount_ = 0;
    ctLastFlatAfterProjectionCount_ = -1;
    ctCorReferenceCaptured_ = acq.value(QStringLiteral("COR_REFERENCE_CAPTURED")).toBool(false);
    ctRunning_ = true;
    ctReplayRunning_ = true;
    ctStopRequested_ = false;
    ctWaitingForFrame_ = false;
    ctWaitingForMotor_ = false;
    ctWaitingForCaptureFinish_ = false;
    ctPendingFrameBatchRemaining_ = 0;
    ctAcquisitionStartedMs_ = QDateTime::currentMSecsSinceEpoch();
    ctFirstFrameRequestedMs_ = 0;
    ctLastFrameEnqueuedMs_ = 0;
    ctWriterFinishRequestedMs_ = 0;
    ctPreparedStartedMs_ = 0;
    ctPreparedReadyMs_ = ctAcquisitionStartedMs_;
    ctCurrentFrameBatchStartedMs_ = 0;
    ctCurrentMotorMoveStartedMs_ = 0;
    ctPreparedSetupMs_ = 0;
    ctMotorMoveMs_ = 0;
    ctRMoveMs_ = 0;
    ctYMoveMs_ = 0;
    ctFrameCaptureMs_ = 0;
    ctFrameEnqueueMs_ = 0;
    ctWriterDrainMs_ = 0;
    ctCompletedMs_ = 0;
    ctMotorMoveCount_ = 0;
    ctRMoveCount_ = 0;
    ctYMoveCount_ = 0;
    ctFrameBatchCount_ = 0;
    ctFrameSavedCount_ = 0;
    ctEstimatedTotalFrames_ = estimateCtTotalFrames();
    ctEstimatedTotalSeconds_ = estimateCtAcquisitionSeconds(ctEstimatedTotalFrames_);

    QDir().mkpath(targetProjectionDir);
    QDir().mkpath(targetOutputDir);
    ctStartButton_->setEnabled(false);
    ctStopButton_->setEnabled(true);
    ctStatusLabel_->setText(QStringLiteral("Replay"));
    if (ctProgressTimer_)
    {
        ctProgressTimer_->start();
    }
    updateCtProgressUi();
    cameraSnapButton_->setEnabled(false);
    cameraLiveButton_->setEnabled(false);
    cameraApplyRoiButton_->setEnabled(false);
    cameraResetRoiButton_->setEnabled(false);
    currentCameraPreviewWindow_ = {};
    cameraPreviewFrameCount_ = 0;
    lastCameraPreviewRenderMs_ = 0;
    cameraPreviewView_->setImage(QPixmap());
    cameraPreviewView_->resetTransform();
    cameraPreviewFitInitialized_ = false;

    ctWriterThreads_ = std::min(std::max(1, ctSettings_.tiffThreads), kAcquisitionWriterThreadLimit);
    ctWriterQueueCapacity_ = kCtFramePoolCapacity;
    ctWriterFrameBytes_ = static_cast<qint64>(std::max(1, ctSettings_.camera.roiWidth))
        * static_cast<qint64>(std::max(1, ctSettings_.camera.roiHeight)) * static_cast<qint64>(2);
    ctWriterStartedMs_ = QDateTime::currentMSecsSinceEpoch();
    ctWriterLabel_->setText(QStringLiteral("%1 threads, 0.0 MiB/s, saved 0").arg(ctWriterThreads_));
    ctBufferLabel_->setText(QStringLiteral("0 / %1 frames, 0.0 / %2 GiB")
        .arg(ctWriterQueueCapacity_)
        .arg(static_cast<double>(ctWriterQueueCapacity_) * static_cast<double>(ctWriterFrameBytes_) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 1));
    QMetaObject::invokeMethod(ctWriter_, [writer = ctWriter_, threads = ctWriterThreads_]() {
        writer->configure(threads, kCtFramePoolCapacity);
        writer->start();
    }, Qt::QueuedConnection);
    writeCtManifest(false, QStringLiteral("Internal CT replay running."));

    ctReplayWorker_ = new CtReplayAcquisitionWorker();
    ctReplayWorker_->moveToThread(&ctReplayThread_);
    connect(ctReplayWorker_, &CtReplayAcquisitionWorker::frameReady, this, &MainWindow::handleReplayCtFrame);
    connect(ctReplayWorker_, &CtReplayAcquisitionWorker::logMessage, this, &MainWindow::appendLog);
    connect(ctReplayWorker_, &CtReplayAcquisitionWorker::replayFinished, this, &MainWindow::finishInternalCtReplay);
    connect(&ctReplayThread_, &QThread::finished, ctReplayWorker_, &QObject::deleteLater);
    ctReplayThread_.start();

    CtReplayAcquisitionSettings replay;
    replay.sourceProjectionDir = sourceProjectionDir;
    replay.sourceJobPath = sourceJob;
    replay.targetProjectionDir = targetProjectionDir;
    replay.targetOutputDir = targetOutputDir;
    replay.tomoPrefix = reconConfig.io.tomoPrefix;
    replay.flatPrefix = reconConfig.io.flatPrefix;
    replay.inputExt = reconConfig.io.inputExt;
    replay.filePattern0 = reconConfig.io.filePattern0;
    replay.filePattern180 = reconConfig.io.filePattern180;
    replay.inputPadding = reconConfig.io.inputPadding;
    replay.projectionCount = reconConfig.geometry.totalAngles;
    replay.imageWidth = reconConfig.geometry.volW;
    replay.imageHeight = reconConfig.geometry.volH;
    replay.frameIntervalMs = qEnvironmentVariableIntValue("CT_REPLAY_FRAME_MS");
    replay.autoCor = acq.value(QStringLiteral("AUTO_COR")).toBool(false);
    replay.corReferenceCaptured = ctCorReferenceCaptured_;
    replay.manifestTemplate = manifest;
    replay.sourceJob = sourceJobObj;
    replay.flatGroups = replayFlatGroups;

    appendLog(QStringLiteral("Internal CT replay autostart: source=%1 target=%2 output=%3 frames=%4 ROI=%5x%6.")
        .arg(sourceProjectionDir)
        .arg(targetProjectionDir)
        .arg(targetOutputDir)
        .arg(replay.projectionCount)
        .arg(replay.imageWidth)
        .arg(replay.imageHeight));
    QMetaObject::invokeMethod(ctReplayWorker_, "startReplay", Qt::QueuedConnection,
        Q_ARG(ct::CtReplayAcquisitionSettings, replay));
}

void MainWindow::handleReplayCtFrame(const CameraFrame& frame, const QString& filePath, bool isFlat, bool isCor180)
{
    if (!ctRunning_ || !ctReplayRunning_)
    {
        return;
    }
    CameraFrame materializedFrame = frame;
    materializedFrame.image = materializeCameraFrameImage(frame);
    if (materializedFrame.image.empty())
    {
        finishInternalCtReplay(false, QStringLiteral("Replay frame has no image buffer."));
        return;
    }
    updateCameraFrameForCtPreview(materializedFrame);
    CtFrameSaveItem item;
    item.filePath = filePath;
    item.bufferLease = materializedFrame.bufferLease;
    item.width = materializedFrame.width;
    item.height = materializedFrame.height;
    item.cvType = materializedFrame.cvType;
    item.strideBytes = materializedFrame.strideBytes;
    if (!item.bufferLease)
    {
        item.ownedImage = std::make_shared<cv::Mat>(materializedFrame.image);
    }
    const qint64 enqueueStartedMs = QDateTime::currentMSecsSinceEpoch();
    ctWriter_->enqueue(item);
    ctFrameEnqueueMs_ += std::max<qint64>(0, QDateTime::currentMSecsSinceEpoch() - enqueueStartedMs);
    ctLastFrameEnqueuedMs_ = QDateTime::currentMSecsSinceEpoch();
    ++ctFrameSavedCount_;
    updateCtProgressUi();
    if (isCor180)
    {
        ctStatusLabel_->setText(QStringLiteral("Replay COR 180"));
        return;
    }
    if (isFlat)
    {
        ++ctFlatIndex_;
        ctStatusLabel_->setText(QStringLiteral("Replay Flat %1").arg(ctFlatIndex_));
        return;
    }
    ++ctProjectionIndex_;
    ctStatusLabel_->setText(QStringLiteral("Replay Tomo %1/%2").arg(ctProjectionIndex_).arg(ctSettings_.projectionCount));
}

void MainWindow::finishInternalCtReplay(bool ok, const QString& message)
{
    appendLog(message);
    if (ctReplayWorker_)
    {
        ctReplayThread_.quit();
        ctReplayThread_.wait();
        ctReplayWorker_ = nullptr;
    }
    if (!ctRunning_ || !ctReplayRunning_)
    {
        return;
    }
    if (!ok)
    {
        ctReplayRunning_ = false;
        QMetaObject::invokeMethod(ctWriter_, "stop", Qt::QueuedConnection);
        finishCtAcquisition(false, message);
        return;
    }
    if (ctWriterFinishRequestedMs_ <= 0)
    {
        ctWriterFinishRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
    }
    QMetaObject::invokeMethod(ctWriter_, "finish", Qt::QueuedConnection);
}

void MainWindow::startCtInitialMotorMove()
{
    if (!ctRunning_ || ctReplayRunning_)
    {
        return;
    }
    ctWaitingForMotor_ = true;
    ctPendingMotorAction_ = QStringLiteral("r_start");
    const double firstTargetAngle = ctSettings_.beginningAngleDeg;
    moveCtAxisAbsolute(QStringLiteral("R"),
        firstTargetAngle,
        ctPendingMotorAction_,
        QStringLiteral("CT R start %1 deg").arg(firstTargetAngle, 0, 'f', 4));
}

void MainWindow::startCtFlyInitialMove()
{
    if (!ctRunning_ || ctReplayRunning_)
    {
        return;
    }
    ctStatusLabel_->setText(QStringLiteral("Fly setup"));
    ctWaitingForMotor_ = true;
    ctPendingMotorAction_ = QStringLiteral("r_start");
    const double firstTargetAngle = ctSettings_.beginningAngleDeg;
    moveCtAxisAbsolute(QStringLiteral("R"),
        firstTargetAngle,
        ctPendingMotorAction_,
        QStringLiteral("CT Fly R start %1 deg").arg(firstTargetAngle, 0, 'f', 4));
}

void MainWindow::advanceCtFlyAcquisition()
{
    if (!ctRunning_ || ctSettings_.mode != CtAcquisitionMode::FlyScan
        || ctWaitingForFrame_ || ctWaitingForCaptureFinish_ || ctWaitingForMotor_)
    {
        return;
    }
    if (ctStopRequested_)
    {
        finishCtAcquisition(false, QStringLiteral("CT fly acquisition stopped."));
        return;
    }
    if (ctSettings_.autoCor && ctProjectionIndex_ >= ctSettings_.projectionCount
        && !ctCorReferenceCaptured_)
    {
        finishCtAcquisition(false, QStringLiteral("CT fly acquisition ended without the in-stream 180 deg COR reference."));
        return;
    }
    if (ctFlatFramesRemaining_ > 0)
    {
        requestCtFrame(CtFrameKind::Flat);
        return;
    }
    if (ctYIsOutForFlat_)
    {
        if (moveCtAxisRelative(QStringLiteral("Y"),
            -ctSettings_.flatMoveDistanceUm,
            QStringLiteral("flat_return"),
            QStringLiteral("CT Fly Y return")))
        {
            return;
        }
        ctYIsOutForFlat_ = false;
    }

    const bool needStartFlat = ctSettings_.flatCount > 0 && ctFlyFlatStage_ == 0;
    const bool needEndFlat = ctSettings_.flatCount > 0 && ctFlyFlatStage_ == 1
        && ctProjectionIndex_ >= ctSettings_.projectionCount && !ctFinalFlatDone_;
    if (needStartFlat || needEndFlat)
    {
        ctFlatFramesRemaining_ = ctSettings_.flatCount;
        ctFlatAfterProjectionCount_ = needStartFlat ? 0 : ctSettings_.projectionCount;
        ctLastFlatAfterProjectionCount_ = ctFlatAfterProjectionCount_;
        if (needStartFlat)
        {
            ctFlyFlatStage_ = 1;
        }
        if (needEndFlat)
        {
            ctFinalFlatDone_ = true;
            ctFlyFlatStage_ = 2;
        }
        CtFlatGroup group;
        group.label = needStartFlat ? QStringLiteral("start") : QStringLiteral("end");
        group.afterProjectionCount = ctFlatAfterProjectionCount_;
        group.firstFlatIndex = ctFlatIndex_;
        group.flatCount = ctSettings_.flatCount;
        group.yMoveUm = ctSettings_.flatMoveDistanceUm;
        ctFlatGroups_.append(group);
        if (moveCtAxisRelative(QStringLiteral("Y"),
            ctSettings_.flatMoveDistanceUm,
            QStringLiteral("flat_out"),
            QStringLiteral("CT Fly Y flat out")))
        {
            return;
        }
        ctYIsOutForFlat_ = true;
        QTimer::singleShot(0, this, &MainWindow::advanceCtFlyAcquisition);
        return;
    }

    if (!ctFlyScanRequested_)
    {
        const double leadStart = ctSettings_.beginningAngleDeg - ctSettings_.flyLeadAngleDeg;
        if (!moveCtAxisAbsolute(QStringLiteral("R"),
            leadStart,
            QStringLiteral("r_fly_start"),
            QStringLiteral("CT Fly R lead start %1 deg").arg(leadStart, 0, 'f', 4)))
        {
            return;
        }
        ctFlyScanRequested_ = true;
        return;
    }

    if (ctProjectionIndex_ >= ctSettings_.projectionCount && (ctFinalFlatDone_ || ctSettings_.flatCount <= 0))
    {
        if (ctWriterFinishRequestedMs_ <= 0)
        {
            ctWriterFinishRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
        }
        QMetaObject::invokeMethod(ctWriter_, "finish", Qt::QueuedConnection);
    }
}

void MainWindow::startCtFlyScanMove()
{
    if (!ctRunning_ || ctSettings_.mode != CtAcquisitionMode::FlyScan || ctFlyScanCapturing_)
    {
        return;
    }
    ctFlyScanCapturing_ = true;
    ctContinuousStartPending_ = true;
    ctContinuousCaptureActive_ = false;
    ctFlyScanMoveDone_ = false;
    ctWaitingForFrame_ = true;
    ctWaitingForCaptureFinish_ = true;
    ctPendingFrameKind_ = CtFrameKind::Tomo;
    ctPendingFrameIsFlat_ = false;
    const int keptFrameCount = ctSettings_.projectionCount + (ctSettings_.autoCor ? 1 : 0);
    ctPendingFrameBatchRemaining_ = keptFrameCount;
    ctFlyContinuousFramesReceived_ = 0;
    ctFlyScanStartedMs_ = QDateTime::currentMSecsSinceEpoch();
    ctStatusLabel_->setText(QStringLiteral("Fly Tomo 0/%1").arg(ctSettings_.projectionCount));

    const double exposureMs = std::max(0.001, exposureMsFromSettings(ctSettings_.camera));

    PvcamCaptureSettings flySettings = ctSettings_.camera;
    flySettings.frameCount = keptFrameCount;
    flySettings.queueCapacity = kCtFramePoolCapacity;
    flySettings.previewIntervalMs = 0;
    flySettings.selectedFrameIndices.clear();
    flySettings.frameIntervalMs = 0.0;
    flySettings.firstKeepDelayMs = ctFlyFirstKeepDelayMs_;
    flySettings.allowFrameIntervalDecimation = false;
    flySettings.rawFrameStride = std::max(1, ctSettings_.rawFrameStride);
    flySettings.framePoolCapacity = kCtFramePoolCapacity;
    flySettings.ringFrameCount = 192;
    flySettings.waitForFrameGate = true;
    flySettings.rawFrameLimit = 0;
    ctFlyRawFramePlan_ = 1 + std::max(0, keptFrameCount - 1) * std::max(1, ctSettings_.rawFrameStride);
    ctCurrentFrameBatchStartedMs_ = QDateTime::currentMSecsSinceEpoch();
    // Preserve frames saved before free-run begins, notably the Fly start-flat group.
    ctContinuousFrameBaseCount_ = ctFrameSavedCount_;
    QMetaObject::invokeMethod(cameraWorker_, "endPreparedCapture", Qt::QueuedConnection);
    QMetaObject::invokeMethod(cameraWorker_, "startContinuousCapture", Qt::QueuedConnection,
        Q_ARG(ct::PvcamCaptureSettings, flySettings));

    appendLog(QStringLiteral("CT fly scan camera arming: speed=%1 deg/s, projections=%2, kept frames=%3, exposure=%4 ms, raw period=%5 ms, stride=%6, projection interval=%7 ms, scan time=%8 s, lead=%9 deg, first keep delay=%10 ms.")
        .arg(ctSettings_.flySpeedDegPerSec, 0, 'f', 4)
        .arg(ctSettings_.projectionCount)
        .arg(keptFrameCount)
        .arg(exposureMs, 0, 'f', 4)
        .arg(ctSettings_.rawFramePeriodMs, 0, 'f', 6)
        .arg(ctSettings_.rawFrameStride)
        .arg(ctSettings_.flyFrameIntervalMs, 0, 'f', 6)
        .arg(ctSettings_.flyScanTimeS, 0, 'f', 3)
        .arg(ctSettings_.flyLeadAngleDeg, 0, 'f', 4)
        .arg(ctFlyFirstKeepDelayMs_, 0, 'f', 3));
}

void MainWindow::startCtFlyScanMotorSweep()
{
    if (!ctRunning_ || ctSettings_.mode != CtAcquisitionMode::FlyScan || !ctFlyScanCapturing_ || ctWaitingForMotor_)
    {
        return;
    }
    const double flyEnd = ctSettings_.beginningAngleDeg + 180.0 + ctSettings_.flyLeadAngleDeg;
    const int speedTable = kCtFlySpeedTable;
    const MotorAxisUi* rAxisUi = nullptr;
    auto rIt = motorAxes_.find(QStringLiteral("R"));
    if (rIt != motorAxes_.end())
    {
        rAxisUi = &(*rIt);
    }
    const double rBaseSpeed = rAxisUi && rAxisUi->baseSpeedSpin
        ? std::min(rAxisUi->baseSpeedSpin->value(), ctSettings_.flySpeedDegPerSec * 0.5)
        : std::min(1.0, ctSettings_.flySpeedDegPerSec * 0.5);
    const double rAccelTimeSec = rAxisUi && rAxisUi->accelSpin
        ? rAxisUi->accelSpin->value()
        : 1.5;
    const MotorSpeedTableParams speedParams = makeMotorSpeedTableParams(
        QStringLiteral("CT Fly R"),
        speedTable,
        ctSettings_.rPulsePerDeg,
        QStringLiteral("deg"),
        rBaseSpeed,
        ctSettings_.flySpeedDegPerSec,
        rAccelTimeSec);
    ctWaitingForMotor_ = true;
    ctPendingMotorAction_ = QStringLiteral("r_fly_scan");
    startCtMotorTiming(ctPendingMotorAction_);
    appendLog(QStringLiteral("CT fly scan motor sweep started: R %1 -> %2 deg; %3.")
        .arg(ctSettings_.beginningAngleDeg - ctSettings_.flyLeadAngleDeg, 0, 'f', 4)
        .arg(flyEnd, 0, 'f', 4)
        .arg(speedParams.summary));
    QMetaObject::invokeMethod(kohzuController_, "configureSpeedTableAndMoveAbsolute", Qt::QueuedConnection,
        Q_ARG(int, ctSettings_.rAxis),
        Q_ARG(qint64, static_cast<qint64>(std::llround(flyEnd * ctSettings_.rPulsePerDeg))),
        Q_ARG(int, speedParams.speedTable),
        Q_ARG(int, speedParams.startSpeedPps),
        Q_ARG(int, speedParams.topSpeedPps),
        Q_ARG(int, speedParams.accelTime10ms),
        Q_ARG(int, speedParams.decelTime10ms),
        Q_ARG(int, speedParams.accelPattern),
        Q_ARG(QString, speedParams.summary),
        Q_ARG(QString, QStringLiteral("CT Fly R scan to %1 deg").arg(flyEnd, 0, 'f', 4)));
}

void MainWindow::startCtTimeSeriesCapture()
{
    if (!ctRunning_ || ctSettings_.mode != CtAcquisitionMode::TimeSeries2D || ctTimeSeriesCapturing_)
    {
        return;
    }
    const double exposureMs = std::max(0.001, exposureMsFromSettings(ctSettings_.camera));
    ctTimeSeriesCapturing_ = true;
    ctContinuousStartPending_ = true;
    ctContinuousCaptureActive_ = false;
    ctWaitingForFrame_ = true;
    ctWaitingForCaptureFinish_ = true;
    ctPendingFrameKind_ = CtFrameKind::Tomo;
    ctPendingFrameBatchRemaining_ = std::max(1, ctSettings_.projectionCount);
    ctProjectionIndex_ = 0;
    ctFirstFrameRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
    ctCurrentFrameBatchStartedMs_ = ctFirstFrameRequestedMs_;
    ctFlyScanStartedMs_ = ctFirstFrameRequestedMs_;
    ctStatusLabel_->setText(QStringLiteral("2D 0/%1").arg(ctSettings_.projectionCount));

    PvcamCaptureSettings settings = ctSettings_.camera;
    settings.frameCount = std::max(1, ctSettings_.projectionCount);
    settings.queueCapacity = kCtFramePoolCapacity;
    settings.previewIntervalMs = 0;
    settings.selectedFrameIndices.clear();
    settings.frameIntervalMs = 0.0;
    settings.firstKeepDelayMs = 0.0;
    settings.allowFrameIntervalDecimation = false;
    settings.rawFrameStride = std::max(1, ctSettings_.rawFrameStride);
    settings.framePoolCapacity = kCtFramePoolCapacity;
    settings.ringFrameCount = 192;
    settings.waitForFrameGate = false;
    settings.rawFrameLimit = 0;
    ctContinuousFrameBaseCount_ = ctFrameSavedCount_;

    appendLog(QStringLiteral("2D time series camera arming: frames=%1, exposure=%2 ms, raw period=%3 ms, raw stride=%4, actual interval=%5 ms, estimated total=%6.")
        .arg(settings.frameCount)
        .arg(exposureMs, 0, 'f', 4)
        .arg(ctSettings_.rawFramePeriodMs, 0, 'f', 6)
        .arg(ctSettings_.rawFrameStride)
        .arg(ctSettings_.timeSeriesFrameIntervalMs, 0, 'f', 6)
        .arg(formatDurationSeconds(ctEstimatedTotalSeconds_)));
    QMetaObject::invokeMethod(cameraWorker_, "startContinuousCapture", Qt::QueuedConnection,
        Q_ARG(ct::PvcamCaptureSettings, settings));
}

void MainWindow::startCtMotorSpeedTableSetup()
{
    if (!ctRunning_ || ctReplayRunning_)
    {
        return;
    }
    if (!kohzuController_ || !motorConnected_)
    {
        finishCtAcquisition(false, QStringLiteral("CT motor speed table setup failed: Kohzu is not connected."));
        return;
    }
    ctMotorSpeedTablesReady_ = false;
    ctMotorSpeedSetupStage_ = 0;
    if (!ctIdlePollingPausedForCt_)
    {
        ctIdlePollingPausedForCt_ = true;
        QMetaObject::invokeMethod(kohzuController_, "setIdlePollingEnabled", Qt::QueuedConnection,
            Q_ARG(bool, false));
    }
    appendLog(ctSettings_.mode == CtAcquisitionMode::FlyScan
        ? QStringLiteral("CT motor setup: preloading R/Y positioning profiles; Fly R profile will be written after lead idle.")
        : QStringLiteral("CT motor setup: preloading R/Y step speed table %1 once for this acquisition.")
            .arg(kCtStepSpeedTable));
    sendNextCtMotorSpeedTableSetup();
}

void MainWindow::sendNextCtMotorSpeedTableSetup()
{
    if (!ctRunning_ || ctReplayRunning_)
    {
        return;
    }
    if (!kohzuController_ || !motorConnected_)
    {
        finishCtAcquisition(false, QStringLiteral("CT motor speed table setup failed: Kohzu is not connected."));
        return;
    }

    QString axisName;
    if (ctMotorSpeedSetupStage_ == 0)
    {
        axisName = QStringLiteral("R");
    }
    else if (ctMotorSpeedSetupStage_ == 1)
    {
        axisName = QStringLiteral("Y");
    }
    else
    {
        finishCtMotorSpeedTableSetup();
        return;
    }

    auto it = motorAxes_.find(axisName);
    if (it == motorAxes_.end())
    {
        finishCtAcquisition(false, QStringLiteral("CT motor speed table setup failed: axis %1 is not configured.").arg(axisName));
        return;
    }

    const double configuredBaseSpeed = it->baseSpeedSpin ? it->baseSpeedSpin->value() : 1.0;
    const double configuredTopSpeed = it->speedSpin ? it->speedSpin->value() : 2.0;
    double configuredAccel = it->accelSpin ? it->accelSpin->value() : 0.5;
    // A normal 1.5 s ramp is appropriate for long positioning moves and for
    // the Fly lead/sweep.  It is pathological for fine step-and-shoot CT:
    // at 1080 projections each R move is only 0.1667 deg, so the axis spends
    // almost the entire move accelerating and decelerating and never reaches
    // the configured top speed.  Use a bounded short ramp only for the fine
    // step profile; Fly and larger step angles retain the configured ramp.
    if (ctSettings_.mode == CtAcquisitionMode::StepAndShoot
        && axisName == QStringLiteral("R")
        && ctSettings_.projectionCount > 0
        && (180.0 / static_cast<double>(ctSettings_.projectionCount)) <= 1.0)
    {
        // 0.50 s is the conservative fine-step setting used by the other
        // positioning axes; it materially reduces the triangular-move time
        // without introducing the very high acceleration of a 0.10 s ramp.
        constexpr double kFineStepAccelSeconds = 0.50;
        configuredAccel = std::min(configuredAccel, kFineStepAccelSeconds);
        appendLog(QStringLiteral("CT fine-step R profile: angle step=%1 deg, accel/decel=%2 s (configured=%3 s).")
            .arg(180.0 / static_cast<double>(ctSettings_.projectionCount), 0, 'f', 6)
            .arg(configuredAccel, 0, 'f', 2)
            .arg(it->accelSpin ? it->accelSpin->value() : 0.5, 0, 'f', 2));
    }
    // All setup/lead/flat moves use the ordinary positioning profile.  The
    // formal Fly R sweep calls configureSpeedTableAndMoveAbsolute() below and
    // performs the one intentional WTB transition to the Fly profile after
    // the lead move has completed and the axis has been confirmed idle.
    const MotorSpeedTableParams speedParams = makeMotorSpeedTableParams(
        QStringLiteral("CT %1").arg(axisName),
        kCtStepSpeedTable,
        it->pulsePerUnit,
        it->unit,
        configuredBaseSpeed,
        configuredTopSpeed,
        configuredAccel);
    ++ctMotorSpeedSetupStage_;
    QMetaObject::invokeMethod(kohzuController_, "configureSpeedTable", Qt::QueuedConnection,
        Q_ARG(int, it->axis),
        Q_ARG(int, speedParams.speedTable),
        Q_ARG(int, speedParams.startSpeedPps),
        Q_ARG(int, speedParams.topSpeedPps),
        Q_ARG(int, speedParams.accelTime10ms),
        Q_ARG(int, speedParams.decelTime10ms),
        Q_ARG(int, speedParams.accelPattern),
        Q_ARG(QString, QStringLiteral("CT speed table %1").arg(axisName)));
}

void MainWindow::finishCtMotorSpeedTableSetup()
{
    if (!ctRunning_ || ctReplayRunning_)
    {
        return;
    }
    ctMotorSpeedTablesReady_ = true;
    appendLog(QStringLiteral("CT motor setup: R/Y speed tables ready; starting Kinetix prepared capture."));
    ctPreparedStartedMs_ = QDateTime::currentMSecsSinceEpoch();
    QMetaObject::invokeMethod(cameraWorker_, "beginPreparedCapture", Qt::QueuedConnection,
        Q_ARG(ct::PvcamCaptureSettings, ctSettings_.camera));
}

void MainWindow::startCtMotorTiming(const QString& action)
{
    Q_UNUSED(action);
    ctCurrentMotorMoveStartedMs_ = QDateTime::currentMSecsSinceEpoch();
}

void MainWindow::finishCtMotorTiming(const QString& action)
{
    if (ctCurrentMotorMoveStartedMs_ <= 0)
    {
        return;
    }
    const qint64 elapsed = std::max<qint64>(0, QDateTime::currentMSecsSinceEpoch() - ctCurrentMotorMoveStartedMs_);
    ctCurrentMotorMoveStartedMs_ = 0;
    ctMotorMoveMs_ += elapsed;
    ++ctMotorMoveCount_;
    if (action.startsWith(QStringLiteral("r_")))
    {
        ctRMoveMs_ += elapsed;
        ++ctRMoveCount_;
    }
    else if (action.startsWith(QStringLiteral("flat_")))
    {
        ctYMoveMs_ += elapsed;
        ++ctYMoveCount_;
    }
}

bool MainWindow::moveCtAxisAbsolute(const QString& axisName, double targetValue, const QString& action, const QString& label)
{
    auto it = motorAxes_.find(axisName);
    if (it == motorAxes_.end())
    {
        finishCtAcquisition(false, QStringLiteral("CT motor move failed: axis %1 is not configured.").arg(axisName));
        return false;
    }
    if (!motorConnected_ || !kohzuController_)
    {
        finishCtAcquisition(false, QStringLiteral("CT motor move failed: Kohzu is not connected."));
        return false;
    }
    if (!ctMotorSpeedTablesReady_)
    {
        finishCtAcquisition(false, QStringLiteral("CT motor move failed: R/Y speed tables are not ready."));
        return false;
    }

    const qint64 pulse = static_cast<qint64>(std::llround(targetValue * it->pulsePerUnit));

    ctWaitingForMotor_ = true;
    ctPendingMotorAction_ = action;
    startCtMotorTiming(action);
    // Initial positioning and the Fly lead move deliberately retain the
    // normal positioning table.  Only startCtFlyScanMotorSweep() changes R
    // to the slower, exposure-defined Fly profile.
    const int speedTable = kCtStepSpeedTable;
    QMetaObject::invokeMethod(kohzuController_, "moveAbsolutePulse", Qt::QueuedConnection,
        Q_ARG(int, it->axis),
        Q_ARG(qint64, pulse),
        Q_ARG(int, speedTable),
        Q_ARG(QString, label));
    return true;
}

bool MainWindow::moveCtAxisRelative(const QString& axisName, double deltaValue, const QString& action, const QString& label)
{
    auto it = motorAxes_.find(axisName);
    if (it == motorAxes_.end())
    {
        finishCtAcquisition(false, QStringLiteral("CT motor move failed: axis %1 is not configured.").arg(axisName));
        return false;
    }
    if (!motorConnected_ || !kohzuController_)
    {
        finishCtAcquisition(false, QStringLiteral("CT motor move failed: Kohzu is not connected."));
        return false;
    }
    if (!ctMotorSpeedTablesReady_)
    {
        finishCtAcquisition(false, QStringLiteral("CT motor move failed: R/Y speed tables are not ready."));
        return false;
    }

    const qint64 pulse = static_cast<qint64>(std::llround(deltaValue * it->pulsePerUnit));
    if (pulse == 0)
    {
        return false;
    }

    ctWaitingForMotor_ = true;
    ctPendingMotorAction_ = action;
    startCtMotorTiming(action);
    const int speedTable = kCtStepSpeedTable;
    QMetaObject::invokeMethod(kohzuController_, "jogRelativePulse", Qt::QueuedConnection,
        Q_ARG(int, it->axis),
        Q_ARG(qint64, pulse),
        Q_ARG(int, speedTable),
        Q_ARG(QString, label));
    return true;
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
        ctCurrentAngleDeg_ = nextAngle;
        moveCtAxisAbsolute(QStringLiteral("R"),
            nextAngle,
            QStringLiteral("r_move"),
            QStringLiteral("CT R to %1 deg").arg(nextAngle, 0, 'f', 4));
        return;
    }
    if (ctStopRequested_)
    {
        finishCtAcquisition(false, QStringLiteral("CT acquisition stopped."));
        return;
    }
    if (ctSettings_.autoCor && ctProjectionIndex_ >= ctSettings_.projectionCount
        && !ctCorReferenceCaptured_)
    {
        ctCorReferencePending_ = true;
        requestCtFrame(CtFrameKind::Cor180);
        return;
    }
    if (ctFlatFramesRemaining_ > 0)
    {
        requestCtFrame(CtFrameKind::Flat);
        return;
    }
    if (ctYIsOutForFlat_)
    {
        if (moveCtAxisRelative(QStringLiteral("Y"),
            -ctSettings_.flatMoveDistanceUm,
            QStringLiteral("flat_return"),
            QStringLiteral("CT Y return")))
        {
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
        if (moveCtAxisRelative(QStringLiteral("Y"),
            ctSettings_.flatMoveDistanceUm,
            QStringLiteral("flat_out"),
            QStringLiteral("CT Y flat out")))
        {
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
    if (ctWriterFinishRequestedMs_ <= 0)
    {
        ctWriterFinishRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
    }
    QMetaObject::invokeMethod(ctWriter_, "finish", Qt::QueuedConnection);
}

void MainWindow::requestCtFrame(CtFrameKind kind)
{
    ctWaitingForFrame_ = true;
    ctWaitingForCaptureFinish_ = true;
    ctPendingFrameKind_ = kind;
    ctPendingFrameIsFlat_ = kind == CtFrameKind::Flat;
    ctPendingFrameBatchRemaining_ = kind == CtFrameKind::Flat
        ? std::max(1, ctFlatFramesRemaining_)
        : 1;
    if (kind == CtFrameKind::Cor180)
    {
        ctStatusLabel_->setText(QStringLiteral("COR 180"));
    }
    else
    {
        ctStatusLabel_->setText(kind == CtFrameKind::Flat
            ? QStringLiteral("Flat %1 x%2").arg(ctFlatIndex_).arg(ctPendingFrameBatchRemaining_)
            : QStringLiteral("Tomo %1/%2").arg(ctProjectionIndex_ + 1).arg(ctSettings_.projectionCount));
    }
    if (ctFirstFrameRequestedMs_ <= 0)
    {
        ctFirstFrameRequestedMs_ = QDateTime::currentMSecsSinceEpoch();
    }
    ctCurrentFrameBatchStartedMs_ = QDateTime::currentMSecsSinceEpoch();
    QMetaObject::invokeMethod(cameraWorker_, "capturePreparedFrame", Qt::QueuedConnection,
        Q_ARG(int, ctPendingFrameBatchRemaining_));
}

void MainWindow::enqueueCtFrame(const CameraFrame& frame)
{
    if (!ctRunning_)
    {
        return;
    }
    CameraFrame materializedFrame = frame;
    materializedFrame.image = materializeCameraFrameImage(frame);
    if (materializedFrame.image.empty())
    {
        finishCtAcquisition(false, QStringLiteral("Kinetix frame has no image buffer."));
        return;
    }
    updateCameraFrameForCtPreview(materializedFrame);
    CtFrameKind kind = ctPendingFrameKind_;
    const bool flyEndpointReference = ctSettings_.mode == CtAcquisitionMode::FlyScan
        && kind == CtFrameKind::Tomo
        && ctSettings_.autoCor
        && frame.logicalIndex == ctSettings_.projectionCount;
    if (flyEndpointReference)
    {
        kind = CtFrameKind::Cor180;
    }
    const bool flat = kind == CtFrameKind::Flat;
    const int fileIndex = flat ? ctFlatIndex_ : ctProjectionIndex_;
    CtFrameSaveItem item;
    const QString savePrefix = ctSettings_.mode == CtAcquisitionMode::TimeSeries2D
        ? ctSettings_.timeSeriesPrefix
        : (flat ? ctSettings_.flatPrefix : ctSettings_.tomoPrefix);
    item.filePath = kind == CtFrameKind::Cor180
        ? QDir(ctSettings_.savePath).filePath(ctSettings_.reconConfig.io.filePattern180)
        : numberedPath(ctSettings_.savePath, savePrefix,
            ctSettings_.padding, fileIndex, ctSettings_.extension);
    item.bufferLease = materializedFrame.bufferLease;
    item.width = materializedFrame.width;
    item.height = materializedFrame.height;
    item.cvType = materializedFrame.cvType;
    item.strideBytes = materializedFrame.strideBytes;
    if (!item.bufferLease)
    {
        item.ownedImage = std::make_shared<cv::Mat>(materializedFrame.image);
    }
    const qint64 enqueueStartedMs = QDateTime::currentMSecsSinceEpoch();
    ctWriter_->enqueue(item);
    ctFrameEnqueueMs_ += std::max<qint64>(0, QDateTime::currentMSecsSinceEpoch() - enqueueStartedMs);
    ctLastFrameEnqueuedMs_ = QDateTime::currentMSecsSinceEpoch();
    ++ctFrameSavedCount_;
    updateCtProgressUi();

    if (kind == CtFrameKind::Cor180)
    {
        QJsonObject corFrame;
        corFrame["RAW_FRAME_INDEX"] = static_cast<double>(frame.frameIndex);
        corFrame["DEVICE_FRAME_NUMBER"] = static_cast<double>(frame.deviceFrameNumber);
        corFrame["BOF_TIMESTAMP_NS"] = static_cast<double>(frame.bofTimestampNs);
        corFrame["EOF_TIMESTAMP_NS"] = static_cast<double>(frame.eofTimestampNs);
        corFrame["TIMING_SOURCE"] = frame.timingSource;
        corFrame["FRAME_INDEX"] = frame.logicalIndex;
        corFrame["LOGICAL_INDEX"] = frame.logicalIndex;
        corFrame["ELAPSED_MS"] = frame.elapsedMs;
        corFrame["TIMESTAMP_MS"] = static_cast<double>(frame.timestampMs);
        corFrame["ESTIMATED_ANGLE_DEG"] = ctSettings_.beginningAngleDeg + 180.0;
        corFrame["FILE"] = ctSettings_.reconConfig.io.filePattern180;
        corFrame["ACQUISITION_STRATEGY"] = flyEndpointReference
            ? QStringLiteral("in_stream_endpoint")
            : QStringLiteral("stationary_post_projection_endpoint");
        if (flyEndpointReference)
        {
            const qint64 expectedFrameOffset = static_cast<qint64>(ctSettings_.projectionCount)
                * static_cast<qint64>(std::max(1, ctSettings_.rawFrameStride));
            corFrame["EXPECTED_RAW_FRAME_OFFSET"] = static_cast<double>(expectedFrameOffset);
            bool endpointVerified = false;
            if (!ctFlyFrames_.isEmpty())
            {
                const QJsonObject& firstFrame = ctFlyFrames_.first();
                const qint64 firstDeviceFrame = static_cast<qint64>(firstFrame.value(QStringLiteral("DEVICE_FRAME_NUMBER")).toDouble());
                const qint64 firstBofNs = static_cast<qint64>(firstFrame.value(QStringLiteral("BOF_TIMESTAMP_NS")).toDouble());
                const qint64 actualFrameOffset = frame.deviceFrameNumber - firstDeviceFrame;
                corFrame["DEVICE_FRAME_OFFSET_FROM_0"] = static_cast<double>(actualFrameOffset);
                corFrame["BOF_OFFSET_MS_FROM_0"] = static_cast<double>(frame.bofTimestampNs - firstBofNs) / 1000000.0;
                endpointVerified = actualFrameOffset == expectedFrameOffset;
            }
            corFrame["ENDPOINT_VERIFIED"] = endpointVerified;
            if (!endpointVerified)
            {
                ctPendingFinishErrorMessage_ = QStringLiteral(
                    "Fly COR 180 endpoint validation failed: expected device-frame offset %1.")
                    .arg(expectedFrameOffset);
                if (cameraWorker_)
                {
                    QMetaObject::invokeMethod(cameraWorker_, "stopCapture", Qt::DirectConnection);
                }
                if (kohzuController_)
                {
                    QMetaObject::invokeMethod(kohzuController_, "stopAllAxes", Qt::QueuedConnection);
                }
            }
            --ctPendingFrameBatchRemaining_;
            ++ctFlyContinuousFramesReceived_;
            appendLog(QStringLiteral("CT Fly COR 180 captured in-stream: device frame %1, logical index %2, elapsed %3 ms.")
                .arg(frame.deviceFrameNumber)
                .arg(frame.logicalIndex)
                .arg(frame.elapsedMs, 0, 'f', 6));
        }
        else
        {
            --ctPendingFrameBatchRemaining_;
            appendLog(QStringLiteral("CT Step COR 180 captured after the formal projection range."));
        }
        ctCor180Frame_ = corFrame;
        ctWaitingForFrame_ = false;
        ctCorReferencePending_ = false;
        ctCorReferenceCaptured_ = true;
        ctStatusLabel_->setText(QStringLiteral("COR 180 saved"));
        return;
    }

    if (flat)
    {
        ++ctFlatIndex_;
        --ctFlatFramesRemaining_;
        --ctPendingFrameBatchRemaining_;
        if (ctFlatFramesRemaining_ <= 0 || ctPendingFrameBatchRemaining_ <= 0)
        {
            ctWaitingForFrame_ = false;
        }
        return;
    }
    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        const int logicalIndex = ctProjectionIndex_;
        const double angleStep = 180.0 / static_cast<double>(std::max(1, ctSettings_.projectionCount));
        QJsonObject flyFrame;
        flyFrame["RAW_FRAME_INDEX"] = static_cast<double>(frame.frameIndex);
        flyFrame["DEVICE_FRAME_NUMBER"] = static_cast<double>(frame.deviceFrameNumber);
        flyFrame["BOF_TIMESTAMP_NS"] = static_cast<double>(frame.bofTimestampNs);
        flyFrame["EOF_TIMESTAMP_NS"] = static_cast<double>(frame.eofTimestampNs);
        flyFrame["TIMING_SOURCE"] = frame.timingSource;
        flyFrame["FRAME_INDEX"] = frame.logicalIndex >= 0 ? static_cast<double>(frame.logicalIndex) : static_cast<double>(logicalIndex);
        flyFrame["LOGICAL_INDEX"] = logicalIndex;
        flyFrame["ELAPSED_MS"] = frame.elapsedMs;
        flyFrame["TIMESTAMP_MS"] = static_cast<double>(frame.timestampMs);
        flyFrame["ESTIMATED_ANGLE_DEG"] = ctSettings_.beginningAngleDeg + static_cast<double>(logicalIndex) * angleStep;
        flyFrame["FILE_INDEX"] = fileIndex;
        ctFlyFrames_.append(flyFrame);
        --ctPendingFrameBatchRemaining_;
        ++ctProjectionIndex_;
        ++ctFlyContinuousFramesReceived_;
        ctStatusLabel_->setText(QStringLiteral("Fly Tomo %1/%2")
            .arg(std::min(ctProjectionIndex_, ctSettings_.projectionCount))
            .arg(ctSettings_.projectionCount));
        if (ctProjectionIndex_ >= ctSettings_.projectionCount && !ctSettings_.autoCor)
        {
            ctWaitingForFrame_ = false;
        }
        return;
    }
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        const int logicalIndex = ctProjectionIndex_;
        QJsonObject frameObj;
        frameObj["RAW_FRAME_INDEX"] = static_cast<double>(frame.frameIndex);
        frameObj["DEVICE_FRAME_NUMBER"] = static_cast<double>(frame.deviceFrameNumber);
        frameObj["BOF_TIMESTAMP_NS"] = static_cast<double>(frame.bofTimestampNs);
        frameObj["EOF_TIMESTAMP_NS"] = static_cast<double>(frame.eofTimestampNs);
        frameObj["TIMING_SOURCE"] = frame.timingSource;
        frameObj["FRAME_INDEX"] = frame.logicalIndex >= 0 ? static_cast<double>(frame.logicalIndex) : static_cast<double>(logicalIndex);
        frameObj["LOGICAL_INDEX"] = logicalIndex;
        frameObj["ELAPSED_MS"] = frame.elapsedMs;
        frameObj["TIMESTAMP_MS"] = static_cast<double>(frame.timestampMs);
        frameObj["FILE_INDEX"] = fileIndex;
        ctTimeSeriesFrames_.append(frameObj);
        --ctPendingFrameBatchRemaining_;
        ++ctProjectionIndex_;
        ctStatusLabel_->setText(QStringLiteral("2D %1/%2")
            .arg(std::min(ctProjectionIndex_, ctSettings_.projectionCount))
            .arg(ctSettings_.projectionCount));
        if (ctProjectionIndex_ >= ctSettings_.projectionCount)
        {
            ctWaitingForFrame_ = false;
        }
        return;
    }
    ctWaitingForFrame_ = false;
    --ctPendingFrameBatchRemaining_;
    ++ctProjectionIndex_;
    ctProjectionNeedsMove_ = true;
}

void MainWindow::handleContinuousFrameBatch(const QVector<ContinuousFrameMetadata>& frames, bool ok, const QString& message)
{
    if (!ctRunning_ || (ctSettings_.mode != CtAcquisitionMode::FlyScan
        && ctSettings_.mode != CtAcquisitionMode::TimeSeries2D))
    {
        return;
    }

    ctFrameSavedCount_ = ctContinuousFrameBaseCount_ + frames.size();
    for (const ContinuousFrameMetadata& metadata : frames)
    {
        if (metadata.cor180)
        {
            QJsonObject corFrame;
            corFrame["RAW_FRAME_INDEX"] = static_cast<double>(metadata.rawFrameIndex);
            corFrame["DEVICE_FRAME_NUMBER"] = static_cast<double>(metadata.deviceFrameNumber);
            corFrame["BOF_TIMESTAMP_NS"] = static_cast<double>(metadata.bofTimestampNs);
            corFrame["EOF_TIMESTAMP_NS"] = static_cast<double>(metadata.eofTimestampNs);
            corFrame["TIMING_SOURCE"] = metadata.timingSource;
            corFrame["FRAME_INDEX"] = metadata.logicalIndex;
            corFrame["LOGICAL_INDEX"] = metadata.logicalIndex;
            corFrame["ELAPSED_MS"] = metadata.elapsedMs;
            corFrame["TIMESTAMP_MS"] = static_cast<double>(metadata.timestampMs);
            corFrame["ESTIMATED_ANGLE_DEG"] = ctSettings_.beginningAngleDeg + 180.0;
            corFrame["FILE"] = ctSettings_.reconConfig.io.filePattern180;
            corFrame["ACQUISITION_STRATEGY"] = QStringLiteral("in_stream_endpoint");
            const qint64 expectedOffset = static_cast<qint64>(ctSettings_.projectionCount)
                * static_cast<qint64>(std::max(1, ctSettings_.rawFrameStride));
            corFrame["EXPECTED_RAW_FRAME_OFFSET"] = static_cast<double>(expectedOffset);
            if (!ctFlyFrames_.isEmpty())
            {
                const QJsonObject& first = ctFlyFrames_.first();
                const qint64 firstDevice = static_cast<qint64>(first.value(QStringLiteral("DEVICE_FRAME_NUMBER")).toDouble());
                const qint64 firstBof = static_cast<qint64>(first.value(QStringLiteral("BOF_TIMESTAMP_NS")).toDouble());
                const qint64 actualOffset = metadata.deviceFrameNumber - firstDevice;
                corFrame["DEVICE_FRAME_OFFSET_FROM_0"] = static_cast<double>(actualOffset);
                corFrame["BOF_OFFSET_MS_FROM_0"] = static_cast<double>(metadata.bofTimestampNs - firstBof) / 1000000.0;
                corFrame["ENDPOINT_VERIFIED"] = actualOffset == expectedOffset;
                if (actualOffset != expectedOffset)
                {
                    ctPendingFinishErrorMessage_ = QStringLiteral(
                        "Fly COR 180 endpoint validation failed: expected device-frame offset %1, got %2.")
                        .arg(expectedOffset)
                        .arg(actualOffset);
                    if (kohzuController_)
                    {
                        QMetaObject::invokeMethod(kohzuController_, "stopAllAxes", Qt::QueuedConnection);
                    }
                }
            }
            ctCor180Frame_ = corFrame;
            ctCorReferenceCaptured_ = true;
            continue;
        }

        if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
        {
            const int logicalIndex = static_cast<int>(ctFlyFrames_.size());
            const double angleStep = 180.0 / static_cast<double>(std::max(1, ctSettings_.projectionCount));
            QJsonObject flyFrame;
            flyFrame["RAW_FRAME_INDEX"] = static_cast<double>(metadata.rawFrameIndex);
            flyFrame["DEVICE_FRAME_NUMBER"] = static_cast<double>(metadata.deviceFrameNumber);
            flyFrame["BOF_TIMESTAMP_NS"] = static_cast<double>(metadata.bofTimestampNs);
            flyFrame["EOF_TIMESTAMP_NS"] = static_cast<double>(metadata.eofTimestampNs);
            flyFrame["TIMING_SOURCE"] = metadata.timingSource;
            flyFrame["FRAME_INDEX"] = metadata.logicalIndex;
            flyFrame["LOGICAL_INDEX"] = logicalIndex;
            flyFrame["ELAPSED_MS"] = metadata.elapsedMs;
            flyFrame["TIMESTAMP_MS"] = static_cast<double>(metadata.timestampMs);
            flyFrame["ESTIMATED_ANGLE_DEG"] = ctSettings_.beginningAngleDeg
                + static_cast<double>(logicalIndex) * angleStep;
            flyFrame["FILE_INDEX"] = metadata.fileIndex;
            ctFlyFrames_.append(flyFrame);
        }
        else
        {
            QJsonObject frameObj;
            frameObj["RAW_FRAME_INDEX"] = static_cast<double>(metadata.rawFrameIndex);
            frameObj["DEVICE_FRAME_NUMBER"] = static_cast<double>(metadata.deviceFrameNumber);
            frameObj["BOF_TIMESTAMP_NS"] = static_cast<double>(metadata.bofTimestampNs);
            frameObj["EOF_TIMESTAMP_NS"] = static_cast<double>(metadata.eofTimestampNs);
            frameObj["TIMING_SOURCE"] = metadata.timingSource;
            frameObj["FRAME_INDEX"] = metadata.logicalIndex;
            frameObj["LOGICAL_INDEX"] = metadata.logicalIndex;
            frameObj["ELAPSED_MS"] = metadata.elapsedMs;
            frameObj["TIMESTAMP_MS"] = static_cast<double>(metadata.timestampMs);
            frameObj["FILE_INDEX"] = metadata.fileIndex;
            ctTimeSeriesFrames_.append(frameObj);
        }
    }

    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        ctProjectionIndex_ = ctFlyFrames_.size();
        ctFlyContinuousFramesReceived_ = frames.size();
        ctStatusLabel_->setText(QStringLiteral("Fly Tomo %1/%2")
            .arg(std::min(ctProjectionIndex_, ctSettings_.projectionCount))
            .arg(ctSettings_.projectionCount));
    }
    else
    {
        ctProjectionIndex_ = ctTimeSeriesFrames_.size();
        ctStatusLabel_->setText(QStringLiteral("2D %1/%2")
            .arg(std::min(ctProjectionIndex_, ctSettings_.projectionCount))
            .arg(ctSettings_.projectionCount));
    }
    updateCtProgressUi();

    if (!ok && !ctStopRequested_ && ctPendingFinishErrorMessage_.isEmpty())
    {
        ctPendingFinishErrorMessage_ = QStringLiteral("Continuous frame router failed: %1").arg(message);
    }
}

void MainWindow::finishCtAcquisition(bool ok, const QString& message)
{
    if (!ctRunning_)
    {
        return;
    }
    ctCompletedMs_ = QDateTime::currentMSecsSinceEpoch();
    if (ctWriterFinishRequestedMs_ > 0)
    {
        ctWriterDrainMs_ += std::max<qint64>(0, ctCompletedMs_ - ctWriterFinishRequestedMs_);
        ctWriterFinishRequestedMs_ = 0;
    }
    if (ctCurrentMotorMoveStartedMs_ > 0)
    {
        finishCtMotorTiming(ctPendingMotorAction_);
    }
    if (ctCurrentFrameBatchStartedMs_ > 0)
    {
        ctFrameCaptureMs_ += std::max<qint64>(0, ctCompletedMs_ - ctCurrentFrameBatchStartedMs_);
        ctCurrentFrameBatchStartedMs_ = 0;
    }
    const bool wasReplay = ctReplayRunning_;
    const bool shouldRestoreLive = ok && ctRestoreLiveAfterFinish_;
    ctRestoreLiveAfterFinish_ = false;
    if (!shouldRestoreLive)
    {
        ++liveRestoreGeneration_;
    }
    ctRunning_ = false;
    const bool motorNeedsStop = !ok && kohzuController_ && ctSettings_.mode != CtAcquisitionMode::TimeSeries2D;
    const QString recoveryState = ok
        ? QStringLiteral("COMPLETED")
        : (motorNeedsStop ? QStringLiteral("STOPPING") : QStringLiteral("STOPPED"));
    writeCtRecoveryState(recoveryState, message);
    ctHardwareStopPending_ = motorNeedsStop;
    ctReplayRunning_ = false;
    ctMotorSpeedTablesReady_ = false;
    ctMotorSpeedSetupStage_ = 0;
    ctWaitingForFrame_ = false;
    ctWaitingForCaptureFinish_ = false;
    ctWaitingForMotor_ = false;
    ctPendingFrameBatchRemaining_ = 0;
    ctContinuousStartPending_ = false;
    ctContinuousCaptureActive_ = false;
    if (!ok && !wasReplay && !ctStopRequested_)
    {
        if (cameraWorker_)
        {
            QMetaObject::invokeMethod(cameraWorker_, "stopCapture", Qt::DirectConnection);
        }
        if (kohzuController_ && ctSettings_.mode != CtAcquisitionMode::TimeSeries2D)
        {
            QMetaObject::invokeMethod(kohzuController_, "stopAllAxes", Qt::QueuedConnection);
        }
    }
    if (!wasReplay && cameraWorker_)
    {
        QMetaObject::invokeMethod(cameraWorker_, "endPreparedCapture", Qt::QueuedConnection);
    }
    if (ctIdlePollingPausedForCt_ && kohzuController_)
    {
        ctIdlePollingPausedForCt_ = false;
        QMetaObject::invokeMethod(kohzuController_, "setIdlePollingEnabled", Qt::QueuedConnection,
            Q_ARG(bool, true));
    }
    if (!ok && ctContinuousTimingStats_.frameLossDetected)
    {
        ctStatusLabel_->setText(QStringLiteral("Failed: device frame lost"));
        ctStatusLabel_->setStyleSheet(QStringLiteral("color:#b00020; font-weight:600;"));
        appendLog(QStringLiteral("FATAL device frame loss: previous=%1, current=%2, missing=%3. Acquisition stopped.")
            .arg(ctContinuousTimingStats_.previousDeviceFrame)
            .arg(ctContinuousTimingStats_.currentDeviceFrame)
            .arg(ctContinuousTimingStats_.missingDeviceFrames));
    }
    else
    {
        ctStatusLabel_->setText(ok ? QStringLiteral("Completed") : QStringLiteral("Failed"));
        ctStatusLabel_->setStyleSheet(ok ? QStringLiteral("color:#1b7f3a;") : QStringLiteral("color:#b00020;"));
    }
    if (ctProgressTimer_)
    {
        ctProgressTimer_->stop();
    }
    updateCtProgressUi();
    if (ok && ctProgressBar_)
    {
        ctProgressBar_->setValue(ctProgressBar_->maximum());
    }
    if (!ok)
    {
        ctWriterStartedMs_ = 0;
    }
    ctStartButton_->setEnabled(true);
    ctStopButton_->setEnabled(false);
    cameraLiveRunning_ = false;
    cameraLiveButton_->setText(QStringLiteral("Open Live"));
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
    if (ctModeTabs_) { ctModeTabs_->setEnabled(true); }
    ctSavePathEdit_->setEnabled(true);
    ctProjectionNumSpin_->setEnabled(true);
    updateCtModeUiState();
    ctTiffThreadsSpin_->setEnabled(true);
    if (ctFlyAllowedBlurSpin_) { ctFlyAllowedBlurSpin_->setEnabled(true); }
    if (ctTimeSeriesRawFrameStrideSpin_) { ctTimeSeriesRawFrameStrideSpin_->setEnabled(true); }
    updateCtModeUiState();
    updateCtFlyDerivedParams();
    writeCtManifest(ok, ok ? QString() : message);
    const qint64 totalMs = ctAcquisitionStartedMs_ > 0
        ? std::max<qint64>(0, ctCompletedMs_ - ctAcquisitionStartedMs_)
        : 0;
    const qint64 motionExposureMs = std::max<qint64>(0, totalMs - ctWriterDrainMs_);
    const QString timingText = QStringLiteral("CT timing: total=%1 s, motion+exposure=%2 s, writer drain=%3 s, frames=%4, motor=%5 s (R=%6 s/%7, Y=%8 s/%9), camera=%10 s/%11 batches, enqueue=%12 s.")
        .arg(formatCtSeconds(totalMs))
        .arg(formatCtSeconds(motionExposureMs))
        .arg(formatCtSeconds(ctWriterDrainMs_))
        .arg(ctFrameSavedCount_)
        .arg(formatCtSeconds(ctMotorMoveMs_))
        .arg(formatCtSeconds(ctRMoveMs_))
        .arg(ctRMoveCount_)
        .arg(formatCtSeconds(ctYMoveMs_))
        .arg(ctYMoveCount_)
        .arg(formatCtSeconds(ctFrameCaptureMs_))
        .arg(ctFrameBatchCount_)
        .arg(formatCtSeconds(ctFrameEnqueueMs_));
    appendLog(timingText);
    appendLog(ok ? QStringLiteral("CT acquisition completed.") : QStringLiteral("CT acquisition failed/stopped: %1").arg(message));
    if (ok && ctSettings_.mode != CtAcquisitionMode::TimeSeries2D)
    {
        enqueueCtReconJob();
        refreshReconQueue();
        startNextQueuedReconstruction();
        if (wasReplay && !ctReplayJobQueue_.isEmpty())
        {
            QTimer::singleShot(250, this, &MainWindow::startInternalCtReplay);
        }
    }
    ctPendingFinishErrorMessage_.clear();
    if (shouldRestoreLive && !shuttingDown_ && !cameraLiveRunning_ && cameraWorker_)
    {
        const int restoreGeneration = ++liveRestoreGeneration_;
        QTimer::singleShot(500, this, [this, restoreGeneration]() {
            if (!shuttingDown_
                && restoreGeneration == liveRestoreGeneration_
                && !cameraLiveRunning_
                && !ctRunning_)
            {
                toggleCameraLivePreview();
            }
        });
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
    root["VERSION"] = 3;
    root["COMPLETED"] = completed;
    root["ERROR"] = errorMessage;
    QString failureCode;
    if (!completed && ctContinuousTimingStats_.frameLossDetected)
    {
        failureCode = QStringLiteral("DeviceFrameLoss");
    }
    else if (!completed && ctStopRequested_)
    {
        failureCode = QStringLiteral("UserStopped");
    }
    else if (!completed && !errorMessage.isEmpty()
        && errorMessage != QStringLiteral("CT acquisition running."))
    {
        failureCode = QStringLiteral("AcquisitionFailed");
    }
    root["FAILURE_CODE"] = failureCode;
    root["CREATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);

    QJsonObject acq;
    const QString modeText = ctSettings_.mode == CtAcquisitionMode::FlyScan
        ? QStringLiteral("FlyScan")
        : (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D ? QStringLiteral("TimeSeries2D") : QStringLiteral("StepAndShoot"));
    acq["MODE"] = modeText;
    acq["BEGINNING_ANGLE_DEG"] = ctSettings_.beginningAngleDeg;
    acq["END_ANGLE_DEG"] = ctSettings_.beginningAngleDeg + 180.0;
    acq["ANGLE_STEP_DEG"] = 180.0 / static_cast<double>(std::max(1, ctSettings_.projectionCount));
    acq["PROJECTION_COUNT"] = ctSettings_.projectionCount;
    acq["FRAME_COUNT"] = ctSettings_.projectionCount;
    acq["TOTAL_ANGLE_DEG"] = 180.0;
    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        acq["FLY_SPEED_DEG_PER_S"] = ctSettings_.flySpeedDegPerSec;
        acq["LEAD_ANGLE_DEG"] = ctSettings_.flyLeadAngleDeg;
        acq["FRAME_INTERVAL_MS"] = ctSettings_.flyFrameIntervalMs;
        acq["RAW_FRAME_PERIOD_MS"] = ctSettings_.rawFramePeriodMs;
        acq["RAW_FRAME_RATE_FPS"] = ctSettings_.rawFrameRateFps;
        acq["RAW_FRAME_STRIDE"] = ctSettings_.rawFrameStride;
        acq["TIMING_SOURCE"] = ctSettings_.timingSource;
        acq["EXPOSURE_MS"] = exposureMsFromSettings(ctSettings_.camera);
        acq["SCAN_TIME_S"] = ctSettings_.flyScanTimeS;
        acq["LEAD_MARGIN_DEG"] = kCtFlyLeadMarginDeg;
        acq["FIRST_KEEP_DELAY_MS"] = ctFlyFirstKeepDelayMs_;
        acq["RAW_FRAME_PLAN"] = ctFlyRawFramePlan_;
        acq["FLY_START_ANGLE_DEG"] = ctSettings_.beginningAngleDeg - ctSettings_.flyLeadAngleDeg;
        acq["FLY_END_ANGLE_DEG"] = ctSettings_.beginningAngleDeg + 180.0 + ctSettings_.flyLeadAngleDeg;
        acq["INTERVAL_STRATEGY"] = QStringLiteral("device_frame_stride");
        acq["ANGLE_MODEL"] = QStringLiteral("constant_motor_speed_device_frame_stride");
    }
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        acq["FRAME_INTERVAL_MS"] = ctSettings_.timeSeriesFrameIntervalMs;
        acq["RAW_FRAME_PERIOD_MS"] = ctSettings_.rawFramePeriodMs;
        acq["RAW_FRAME_RATE_FPS"] = ctSettings_.rawFrameRateFps;
        acq["RAW_FRAME_STRIDE"] = ctSettings_.rawFrameStride;
        acq["TIMING_SOURCE"] = ctSettings_.timingSource;
        acq["EXPOSURE_MS"] = exposureMsFromSettings(ctSettings_.camera);
        acq["INTERVAL_STRATEGY"] = ctContinuousIntervalStrategy_;
        acq["INTERVAL_FALLBACK_REASON"] = ctContinuousIntervalFallbackReason_;
        acq["SAVED_FRAME_NUMBER_MODEL"] = QStringLiteral("first + k*RAW_FRAME_STRIDE");
        acq["ANGLE_MODEL"] = QStringLiteral("none_fixed_position_time_series");
    }
    acq["FLAT_GAP"] = ctSettings_.flatGap;
    acq["FLAT_MOVE_DISTANCE_UM"] = ctSettings_.flatMoveDistanceUm;
    acq["FLAT_NUM"] = ctSettings_.flatCount;
    acq["IMAGE_WIDTH"] = ctSettings_.camera.roiWidth;
    acq["IMAGE_HEIGHT"] = ctSettings_.camera.roiHeight;
    acq["AUTO_COR"] = ctSettings_.autoCor;
    acq["COR_REFERENCE_CAPTURED"] = ctCorReferenceCaptured_;
    acq["COR_REFERENCE_0_FILE"] = ctSettings_.reconConfig.io.filePattern0;
    acq["COR_REFERENCE_180_FILE"] = ctSettings_.reconConfig.io.filePattern180;
    acq["COR_REFERENCE_0_SOURCE"] = QStringLiteral("first_formal_projection");
    acq["COR_REFERENCE_180_STRATEGY"] = ctSettings_.mode == CtAcquisitionMode::FlyScan
        ? QStringLiteral("in_stream_endpoint")
        : QStringLiteral("stationary_post_projection_endpoint");
    acq["FORMAL_PROJECTION_RANGE"] = QStringLiteral("[begin, begin+180)");
    acq["CONTINUOUS_KEPT_FRAME_COUNT"] = ctSettings_.mode == CtAcquisitionMode::FlyScan
        ? ctSettings_.projectionCount + (ctSettings_.autoCor ? 1 : 0)
        : 0;
    root["ACQUISITION"] = acq;
    root["CAMERA"] = pvcamSettingsToJson(ctSettings_.camera);
    if (ctSettings_.mode == CtAcquisitionMode::FlyScan
        || ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        QJsonObject calibration;
        calibration["STATUS"] = timingCalibrationResult_.ok ? QStringLiteral("ready") : QStringLiteral("invalid");
        calibration["SOURCE"] = timingCalibrationResult_.timingSource;
        calibration["METADATA_VERSION"] = timingCalibrationResult_.metadataVersion;
        calibration["SAMPLE_COUNT"] = timingCalibrationResult_.analysis.sampleCount;
        calibration["FIRST_DEVICE_FRAME"] = static_cast<double>(timingCalibrationResult_.analysis.firstFrameNumber);
        calibration["LAST_DEVICE_FRAME"] = static_cast<double>(timingCalibrationResult_.analysis.lastFrameNumber);
        calibration["DROPPED_FRAMES"] = static_cast<double>(timingCalibrationResult_.analysis.droppedFrames);
        calibration["MEAN_PERIOD_MS"] = timingCalibrationResult_.analysis.meanPeriodMs;
        calibration["MEDIAN_PERIOD_MS"] = timingCalibrationResult_.analysis.medianPeriodMs;
        calibration["FITTED_PERIOD_MS"] = timingCalibrationResult_.analysis.fittedPeriodMs;
        calibration["MIN_PERIOD_MS"] = timingCalibrationResult_.analysis.minPeriodMs;
        calibration["MAX_PERIOD_MS"] = timingCalibrationResult_.analysis.maxPeriodMs;
        calibration["STDDEV_MS"] = timingCalibrationResult_.analysis.standardDeviationMs;
        calibration["MAX_PERIOD_DEVIATION_MS"] = timingCalibrationResult_.analysis.maxPeriodDeviationMs;
        calibration["FIRST_HALF_MEAN_PERIOD_MS"] = timingCalibrationResult_.analysis.firstHalfMeanPeriodMs;
        calibration["SECOND_HALF_MEAN_PERIOD_MS"] = timingCalibrationResult_.analysis.secondHalfMeanPeriodMs;
        calibration["MEAN_DRIFT_MS"] = timingCalibrationResult_.analysis.meanDriftMs;
        calibration["MEAN_DRIFT_PERCENT"] = timingCalibrationResult_.analysis.meanDriftPercent;
        calibration["MAX_FIT_RESIDUAL_MS"] = timingCalibrationResult_.analysis.maxFitResidualMs;
        calibration["EXPOSURE_TIMING_AVAILABLE"] = timingCalibrationResult_.analysis.exposureTimingAvailable;
        calibration["MEAN_EXPOSURE_MS"] = timingCalibrationResult_.analysis.meanExposureMs;
        calibration["MIN_EXPOSURE_MS"] = timingCalibrationResult_.analysis.minExposureMs;
        calibration["MAX_EXPOSURE_MS"] = timingCalibrationResult_.analysis.maxExposureMs;
        calibration["MAX_EXPOSURE_DEVIATION_MS"] = timingCalibrationResult_.analysis.maxExposureDeviationMs;
        calibration["RAW_FRAME_RATE_FPS"] = timingCalibrationResult_.analysis.rawFrameRateFps;
        root["TIMING_CALIBRATION"] = calibration;
    }

    QJsonObject io;
    io["TOMO_PREFIX"] = ctSettings_.tomoPrefix;
    io["FLAT_PREFIX"] = ctSettings_.flatPrefix;
    io["TIME_SERIES_PREFIX"] = ctSettings_.timeSeriesPrefix;
    io["DARK_PREFIX"] = ctSettings_.reconConfig.io.darkPrefix;
    io["INPUT_PADDING"] = ctSettings_.padding;
    io["INPUT_EXT"] = ctSettings_.extension;
    io["FILE_PATTERN_0"] = ctSettings_.reconConfig.io.filePattern0;
    io["FILE_PATTERN_180"] = ctSettings_.reconConfig.io.filePattern180;
    root["IO"] = io;
    root["FLAT_GROUPS"] = flatGroupsToJson(ctFlatGroups_);
    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        QJsonArray frames;
        for (const QJsonObject& frame : ctFlyFrames_)
        {
            frames.append(frame);
        }
        root["FLY_FRAMES"] = frames;
    }
    if (!ctCor180Frame_.isEmpty())
    {
        root["COR_REFERENCE_180_FRAME"] = ctCor180Frame_;
    }
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        QJsonArray frames;
        for (const QJsonObject& frame : ctTimeSeriesFrames_)
        {
            frames.append(frame);
        }
        root["TIME_SERIES_FRAMES"] = frames;
    }

    QJsonObject motor;
    motor["R_AXIS"] = ctSettings_.rAxis;
    motor["R_PULSE_PER_DEG"] = ctSettings_.rPulsePerDeg;
    motor["Y_AXIS"] = ctSettings_.yAxis;
    motor["Y_PULSE_PER_UM"] = ctSettings_.yPulsePerUm;
    root["MOTOR"] = motor;

    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    const qint64 effectiveEndMs = ctCompletedMs_ > 0 ? ctCompletedMs_ : nowMs;
    const qint64 totalMs = ctAcquisitionStartedMs_ > 0
        ? std::max<qint64>(0, effectiveEndMs - ctAcquisitionStartedMs_)
        : 0;
    const qint64 pendingMotorMs = ctCurrentMotorMoveStartedMs_ > 0
        ? std::max<qint64>(0, nowMs - ctCurrentMotorMoveStartedMs_)
        : 0;
    const qint64 pendingFrameBatchMs = ctCurrentFrameBatchStartedMs_ > 0
        ? std::max<qint64>(0, nowMs - ctCurrentFrameBatchStartedMs_)
        : 0;
    const qint64 writerDrainMs = ctWriterDrainMs_ + ((ctWriterFinishRequestedMs_ > 0)
        ? std::max<qint64>(0, nowMs - ctWriterFinishRequestedMs_)
        : 0);
    const qint64 motorMs = ctMotorMoveMs_ + pendingMotorMs;
    const qint64 frameCaptureMs = ctFrameCaptureMs_ + pendingFrameBatchMs;
    const qint64 motionExposureMs = std::max<qint64>(0, totalMs - writerDrainMs);
    QJsonObject timing;
    timing["STARTED_AT"] = ctAcquisitionStartedMs_ > 0
        ? QDateTime::fromMSecsSinceEpoch(ctAcquisitionStartedMs_).toString(Qt::ISODate)
        : QString();
    timing["FINISHED_AT"] = completed
        ? QDateTime::fromMSecsSinceEpoch(effectiveEndMs).toString(Qt::ISODate)
        : QString();
    addTimingMs(timing, QStringLiteral("TOTAL"), totalMs);
    addTimingMs(timing, QStringLiteral("MOTION_EXPOSURE"), motionExposureMs);
    addTimingMs(timing, QStringLiteral("PREPARED_SETUP"), ctPreparedSetupMs_);
    addTimingMs(timing, QStringLiteral("MOTOR_TOTAL"), motorMs);
    addTimingMs(timing, QStringLiteral("R_MOTOR"), ctRMoveMs_);
    addTimingMs(timing, QStringLiteral("Y_MOTOR"), ctYMoveMs_);
    addTimingMs(timing, QStringLiteral("CAMERA_CAPTURE"), frameCaptureMs);
    addTimingMs(timing, QStringLiteral("FRAME_ENQUEUE"), ctFrameEnqueueMs_);
    addTimingMs(timing, QStringLiteral("WRITER_DRAIN"), writerDrainMs);
    timing["FRAME_COUNT"] = ctFrameSavedCount_;
    timing["TOMO_COUNT_SAVED"] = ctProjectionIndex_;
    timing["FLAT_COUNT_SAVED"] = ctFlatIndex_;
    timing["COR180_SAVED"] = ctCorReferenceCaptured_;
    timing["CONTINUOUS_RAW_FRAMES"] = static_cast<double>(ctContinuousRawFrames_);
    timing["CONTINUOUS_DISCARDED_FRAMES"] = static_cast<double>(ctContinuousDiscardedFrames_);
    timing["CONTINUOUS_DEVICE_DROPPED_FRAMES"] = static_cast<double>(ctContinuousDeviceDroppedFrames_);
    timing["CONTINUOUS_DUPLICATE_NOTIFICATIONS"] = static_cast<double>(ctContinuousTimingStats_.duplicateNotifications);
    timing["RAW_BASELINE_PERIOD_MS"] = ctContinuousTimingStats_.baselinePeriodMs;
    timing["RAW_OVERALL_PERIOD_MS"] = ctContinuousTimingStats_.overallPeriodMs;
    timing["RAW_OVERALL_FRAME_RATE_FPS"] = ctContinuousTimingStats_.overallFrameRateFps;
    timing["RAW_ROLLING_PERIOD_FINAL_MS"] = ctContinuousTimingStats_.rollingPeriodMs;
    timing["RAW_ROLLING_FRAME_RATE_FINAL_FPS"] = ctContinuousTimingStats_.rollingFrameRateFps;
    timing["RAW_ROLLING_DRIFT_FINAL_PERCENT"] = ctContinuousTimingStats_.rollingDriftPercent;
    timing["RAW_MIN_INTERVAL_MS"] = ctContinuousTimingStats_.minIntervalMs;
    timing["RAW_MAX_INTERVAL_MS"] = ctContinuousTimingStats_.maxIntervalMs;
    timing["RAW_MAX_INTERVAL_DEVIATION_MS"] = ctContinuousTimingStats_.maxIntervalDeviationMs;
    timing["RAW_FINAL_PHASE_ERROR_MS"] = ctContinuousTimingStats_.currentPhaseErrorMs;
    timing["RAW_MAX_ABS_PHASE_ERROR_MS"] = ctContinuousTimingStats_.maxAbsPhaseErrorMs;
    timing["RAW_FITTED_PERIOD_MS"] = ctContinuousTimingStats_.fittedPeriodMs;
    timing["RAW_FIRST_HALF_MEAN_PERIOD_MS"] = ctContinuousTimingStats_.firstHalfMeanPeriodMs;
    timing["RAW_SECOND_HALF_MEAN_PERIOD_MS"] = ctContinuousTimingStats_.secondHalfMeanPeriodMs;
    timing["RAW_MEAN_DRIFT_MS"] = ctContinuousTimingStats_.meanDriftMs;
    timing["RAW_MEAN_DRIFT_PERCENT"] = ctContinuousTimingStats_.meanDriftPercent;
    timing["RAW_MAX_FIT_RESIDUAL_MS"] = ctContinuousTimingStats_.maxFitResidualMs;
    timing["DEVICE_FRAME_LOSS_DETECTED"] = ctContinuousTimingStats_.frameLossDetected;
    timing["DEVICE_FRAME_LOSS_PREVIOUS"] = static_cast<double>(ctContinuousTimingStats_.previousDeviceFrame);
    timing["DEVICE_FRAME_LOSS_CURRENT"] = static_cast<double>(ctContinuousTimingStats_.currentDeviceFrame);
    timing["DEVICE_FRAME_LOSS_MISSING"] = static_cast<double>(ctContinuousTimingStats_.missingDeviceFrames);
    timing["FLY_MAX_ANGLE_TIMING_ERROR_DEG"] = ctSettings_.mode == CtAcquisitionMode::FlyScan
        ? ctSettings_.flySpeedDegPerSec * ctContinuousTimingStats_.maxAbsPhaseErrorMs / 1000.0
        : 0.0;
    timing["FRAME_POOL_CAPACITY"] = kCtFramePoolCapacity;
    timing["FRAME_POOL_PEAK"] = ctFramePoolPeak_;
    timing["PVCAM_RING_CAPACITY"] = ctPvcamRingCapacity_;
    timing["PVCAM_RING_ESTIMATED_UNREAD_FINAL"] = ctPvcamRingEstimatedUnread_;
    timing["PVCAM_RING_ESTIMATED_UNREAD_PEAK"] = ctPvcamRingPeakEstimatedUnread_;
    if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
    {
        timing["FLY_RAW_FRAME_PLAN"] = ctFlyRawFramePlan_;
        timing["FLY_KEPT_TOMO_FRAMES"] = ctFlyFrames_.size();
        timing["FLY_KEPT_TOTAL_FRAMES"] = ctFlyFrames_.size() + (ctCorReferenceCaptured_ ? 1 : 0);
        timing["FLY_COR180_IN_STREAM"] = ctCorReferenceCaptured_;
        timing["FLY_FRAME_INTERVAL_MS"] = ctSettings_.flyFrameIntervalMs;
        timing["FLY_SCAN_TIME_S"] = ctSettings_.flyScanTimeS;
        timing["FLY_SCAN_STARTED_AT"] = ctFlyScanStartedMs_ > 0
            ? QDateTime::fromMSecsSinceEpoch(ctFlyScanStartedMs_).toString(Qt::ISODate)
            : QString();
        timing["FLY_SCAN_FINISHED_AT"] = ctFlyScanFinishedMs_ > 0
            ? QDateTime::fromMSecsSinceEpoch(ctFlyScanFinishedMs_).toString(Qt::ISODate)
            : QString();
    }
    if (ctSettings_.mode == CtAcquisitionMode::TimeSeries2D)
    {
        timing["TIME_SERIES_FRAME_INTERVAL_MS"] = ctSettings_.timeSeriesFrameIntervalMs;
        timing["TIME_SERIES_SAVED_FRAMES"] = ctTimeSeriesFrames_.size();
        timing["INTERVAL_STRATEGY"] = ctContinuousIntervalStrategy_;
        timing["INTERVAL_FALLBACK_REASON"] = ctContinuousIntervalFallbackReason_;
        timing["TIME_SERIES_STARTED_AT"] = ctFlyScanStartedMs_ > 0
            ? QDateTime::fromMSecsSinceEpoch(ctFlyScanStartedMs_).toString(Qt::ISODate)
            : QString();
        timing["TIME_SERIES_FINISHED_AT"] = ctFlyScanFinishedMs_ > 0
            ? QDateTime::fromMSecsSinceEpoch(ctFlyScanFinishedMs_).toString(Qt::ISODate)
            : QString();
    }
    timing["FRAME_BATCH_COUNT"] = ctFrameBatchCount_;
    timing["MOTOR_MOVE_COUNT"] = ctMotorMoveCount_;
    timing["R_MOVE_COUNT"] = ctRMoveCount_;
    timing["Y_MOVE_COUNT"] = ctYMoveCount_;
    timing["AVG_FRAME_CAPTURE_MS"] = ctFrameSavedCount_ > 0
        ? static_cast<double>(frameCaptureMs) / static_cast<double>(ctFrameSavedCount_)
        : 0.0;
    timing["AVG_MOTOR_MOVE_MS"] = ctMotorMoveCount_ > 0
        ? static_cast<double>(motorMs) / static_cast<double>(ctMotorMoveCount_)
        : 0.0;
    timing["AVG_R_MOVE_MS"] = ctRMoveCount_ > 0
        ? static_cast<double>(ctRMoveMs_) / static_cast<double>(ctRMoveCount_)
        : 0.0;
    timing["AVG_Y_MOVE_MS"] = ctYMoveCount_ > 0
        ? static_cast<double>(ctYMoveMs_) / static_cast<double>(ctYMoveCount_)
        : 0.0;
    timing["WRITER_THREADS"] = ctWriterThreads_;
    timing["WRITER_QUEUE_CAPACITY"] = ctWriterQueueCapacity_;
    timing["FRAME_BYTES"] = static_cast<double>(ctWriterFrameBytes_);
    timing["WRITTEN_BYTES_ESTIMATE"] = static_cast<double>(std::max(0, ctFrameSavedCount_))
        * static_cast<double>(std::max<qint64>(0, ctWriterFrameBytes_));
    timing["AVG_WRITE_MIB_PER_S"] = totalMs > 0
        ? (static_cast<double>(std::max(0, ctFrameSavedCount_)) * static_cast<double>(std::max<qint64>(0, ctWriterFrameBytes_)))
            / (1024.0 * 1024.0) / (static_cast<double>(totalMs) / 1000.0)
        : 0.0;
    root["TIMING"] = timing;
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
    job["JOB_SOURCE"] = "CT_ACQUISITION";
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

void MainWindow::enqueueCurrentReconJob()
{
    AppConfig cfg = configFromUi();
    if (cfg.io.reconMethod == ReconMethod::RealTimeFBP)
    {
        cfg.io.reconMethod = ReconMethod::OfflineFBP;
    }

    const QString projectionDir = QDir::fromNativeSeparators(cfg.io.watchDir.trimmed());
    if (projectionDir.isEmpty())
    {
        appendLog(QStringLiteral("Queue Current Recon failed: Watch Dir is empty."));
        return;
    }
    if (!QDir(projectionDir).exists())
    {
        appendLog(QStringLiteral("Queue Current Recon failed: Watch Dir does not exist: %1").arg(projectionDir));
        return;
    }

    if (cfg.io.saveDir.trimmed().isEmpty())
    {
        cfg.io.saveDir = QDir(projectionDir).filePath(QStringLiteral("recon"));
    }
    cfg.io.watchDir = projectionDir;
    cfg.io.saveDir = QDir::fromNativeSeparators(cfg.io.saveDir);
    if (!QDir().mkpath(cfg.io.saveDir))
    {
        appendLog(QStringLiteral("Queue Current Recon failed: cannot create Save Dir: %1").arg(cfg.io.saveDir));
        return;
    }

    if (cfg.geometry.totalAngles <= 0 || cfg.geometry.volW <= 0 || cfg.geometry.volH <= 0)
    {
        appendLog(QStringLiteral("Queue Current Recon failed: invalid geometry or total angle count."));
        return;
    }

    const QString originalFilePattern0 = cfg.io.filePattern0;
    const int tomoStartIndex = resolveTomoStartIndexForConfig(cfg);
    if (cfg.io.filePattern0 != originalFilePattern0)
    {
        appendLog(QStringLiteral("Queue Current Recon adjusted FILE_PATTERN_0 to existing first Tomo: %1")
            .arg(cfg.io.filePattern0));
    }
    const QString firstTomo = QDir(projectionDir).filePath(numberedFileName(cfg.io.tomoPrefix,
        cfg.io.inputPadding, tomoStartIndex, cfg.io.inputExt));
    const QString lastTomo = QDir(projectionDir).filePath(numberedFileName(cfg.io.tomoPrefix,
        cfg.io.inputPadding, tomoStartIndex + cfg.geometry.totalAngles - 1, cfg.io.inputExt));
    if (!QFileInfo::exists(firstTomo))
    {
        appendLog(QStringLiteral("Queue Current Recon failed: missing first Tomo file: %1").arg(firstTomo));
        return;
    }
    if (!QFileInfo::exists(lastTomo))
    {
        appendLog(QStringLiteral("Queue Current Recon failed: missing last Tomo file: %1").arg(lastTomo));
        return;
    }

    QVector<FlatGroupConfig> flatGroups;
    if (cfg.io.correctionMode == CorrectionMode::Flat || cfg.io.correctionMode == CorrectionMode::FlatDark)
    {
        const int firstFlatIndex = findFirstFlatIndexForConfig(cfg);
        if (firstFlatIndex < 0)
        {
            appendLog(QStringLiteral("Queue Current Recon failed: no flat file matching prefix %1 in %2.")
                .arg(cfg.io.flatPrefix, projectionDir));
            return;
        }
        flatGroups = inferQueueFlatGroups(cfg, firstFlatIndex);
        for (const FlatGroupConfig& group : flatGroups)
        {
            const QString firstFlat = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
                cfg.io.inputPadding, group.firstFlatIndex, cfg.io.inputExt));
            const QString lastFlat = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
                cfg.io.inputPadding, group.firstFlatIndex + group.flatCount - 1, cfg.io.inputExt));
            if (!QFileInfo::exists(firstFlat))
            {
                appendLog(QStringLiteral("Queue Current Recon failed: missing flat file: %1").arg(firstFlat));
                return;
            }
            if (!QFileInfo::exists(lastFlat))
            {
                appendLog(QStringLiteral("Queue Current Recon failed: missing flat file: %1").arg(lastFlat));
                return;
            }
        }
    }
    cfg.preprocess.flatGroups = flatGroups;

    if (cfg.geometry.corAuto)
    {
        const QString ref0 = QDir(projectionDir).filePath(cfg.io.filePattern0);
        const QString ref180 = QDir(projectionDir).filePath(cfg.io.filePattern180);
        if (!QFileInfo::exists(ref0))
        {
            appendLog(QStringLiteral("Queue Current Recon failed: missing Auto COR 0 deg reference: %1").arg(ref0));
            return;
        }
        if (!QFileInfo::exists(ref180))
        {
            appendLog(QStringLiteral("Queue Current Recon failed: missing Auto COR 180 deg reference: %1").arg(ref180));
            return;
        }
    }

    QJsonObject manifest;
    manifest["TYPE"] = "CT_STREAM_SCAN_MANIFEST";
    manifest["VERSION"] = 2;
    manifest["COMPLETED"] = true;
    manifest["ERROR"] = QString();
    manifest["CREATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);

    QJsonObject acq;
    acq["BEGINNING_ANGLE_DEG"] = cfg.geometry.angleOffsetDeg;
    acq["END_ANGLE_DEG"] = static_cast<double>(cfg.geometry.angleOffsetDeg) + 180.0;
    acq["ANGLE_STEP_DEG"] = 180.0 / static_cast<double>(std::max(1, cfg.geometry.totalAngles));
    acq["PROJECTION_COUNT"] = cfg.geometry.totalAngles;
    acq["FLAT_GAP"] = cfg.preprocess.flatGap;
    acq["FLAT_MOVE_DISTANCE_UM"] = 0.0;
    acq["FLAT_NUM"] = cfg.preprocess.flatNum;
    acq["IMAGE_WIDTH"] = cfg.geometry.volW;
    acq["IMAGE_HEIGHT"] = cfg.geometry.volH;
    acq["AUTO_COR"] = cfg.geometry.corAuto;
    acq["COR_REFERENCE_CAPTURED"] = cfg.geometry.corAuto;
    acq["COR_REFERENCE_0_FILE"] = cfg.io.filePattern0;
    acq["COR_REFERENCE_180_FILE"] = cfg.io.filePattern180;
    manifest["ACQUISITION"] = acq;

    QJsonObject io;
    io["TOMO_PREFIX"] = cfg.io.tomoPrefix;
    io["FLAT_PREFIX"] = cfg.io.flatPrefix;
    io["DARK_PREFIX"] = cfg.io.darkPrefix;
    io["INPUT_PADDING"] = cfg.io.inputPadding;
    io["INPUT_EXT"] = cfg.io.inputExt;
    io["FILE_PATTERN_0"] = cfg.io.filePattern0;
    io["FILE_PATTERN_180"] = cfg.io.filePattern180;
    manifest["IO"] = io;
    manifest["FLAT_GROUPS"] = flatGroupConfigsToJson(flatGroups);

    QJsonObject camera;
    camera["ROI_X"] = 0;
    camera["ROI_Y"] = 0;
    camera["ROI_WIDTH"] = cfg.geometry.volW;
    camera["ROI_HEIGHT"] = cfg.geometry.volH;
    manifest["CAMERA"] = camera;

    const QString manifestPath = manifestPathForProjectionDir(projectionDir);
    if (!writeJsonFileAtomic(manifestPath, manifest))
    {
        appendLog(QStringLiteral("Queue Current Recon failed: cannot write manifest: %1").arg(manifestPath));
        return;
    }

    const QString pendingDir = QDir(reconQueueRoot()).filePath(QStringLiteral("pending"));
    QDir().mkpath(pendingDir);
    const QStringList existing = QDir(pendingDir).entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
    int maxIndex = 0;
    for (const QString& fileName : existing)
    {
        maxIndex = std::max(maxIndex, queueIndexFromFileName(fileName));
    }
    if (maxIndex == std::numeric_limits<int>::max())
    {
        maxIndex = existing.size() * 10;
    }
    const int queueIndex = maxIndex + 10;
    QString base = sanitizeQueueName(QFileInfo(projectionDir).fileName());
    if (base.isEmpty())
    {
        base = QStringLiteral("manual_recon");
    }
    const QString jobPath = QDir(pendingDir).filePath(queueFileName(queueIndex, base));
    QJsonObject job = appConfigToJobObject(cfg);
    job["TYPE"] = "CT_STREAM_RECON_JOB";
    job["VERSION"] = 2;
    job["STATUS"] = "pending";
    job["JOB_SOURCE"] = "MANUAL_QUEUE";
    job["QUEUE_INDEX"] = queueIndex;
    job["CREATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    job["PROJECTION_DIR"] = projectionDir;
    job["OUTPUT_DIR"] = cfg.io.saveDir;

    if (!writeJsonFileAtomic(jobPath, job))
    {
        appendLog(QStringLiteral("Queue Current Recon failed: cannot write job: %1").arg(jobPath));
        return;
    }

    appendLog(QStringLiteral("Current reconstruction queued: manifest=%1, job=%2").arg(manifestPath, jobPath));
    refreshReconQueue();
    startNextQueuedReconstruction();
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

void MainWindow::importReconJobs()
{
    const QStringList sourcePaths = QFileDialog::getOpenFileNames(this,
        QStringLiteral("Import Recon Job JSON"),
        QDir::homePath(),
        QStringLiteral("JSON Files (*.json);;All Files (*)"));
    if (sourcePaths.isEmpty())
    {
        return;
    }

    const QString pendingDirPath = QDir(reconQueueRoot()).filePath(QStringLiteral("pending"));
    QDir().mkpath(pendingDirPath);
    QDir pendingDir(pendingDirPath);
    normalizePendingReconQueue(false);

    const QStringList existing = pendingDir.entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
    int maxIndex = 0;
    for (const QString& fileName : existing)
    {
        const int index = queueIndexFromFileName(fileName);
        if (index != std::numeric_limits<int>::max())
        {
            maxIndex = std::max(maxIndex, index);
        }
    }

    int imported = 0;
    int skipped = 0;
    for (const QString& sourcePath : sourcePaths)
    {
        QJsonObject job;
        QString error;
        if (!readJsonFile(sourcePath, &job, &error))
        {
            appendLog(QStringLiteral("Import Job skipped: %1").arg(error));
            ++skipped;
            continue;
        }

        QString projectionDir = QDir::fromNativeSeparators(projectionDirFromJob(job).trimmed());
        if (projectionDir.isEmpty())
        {
            appendLog(QStringLiteral("Import Job skipped: missing PROJECTION_DIR in %1").arg(sourcePath));
            ++skipped;
            continue;
        }
        if (!QDir(projectionDir).exists())
        {
            appendLog(QStringLiteral("Import Job skipped: PROJECTION_DIR does not exist: %1").arg(projectionDir));
            ++skipped;
            continue;
        }

        const QString manifestPath = manifestPathForProjectionDir(projectionDir);
        QJsonObject manifest;
        if (!readJsonFile(manifestPath, &manifest, &error))
        {
            appendLog(QStringLiteral("Import Job skipped: missing or invalid scan_manifest.json for %1: %2")
                .arg(projectionDir, error));
            ++skipped;
            continue;
        }
        if (!manifest.value(QStringLiteral("COMPLETED")).toBool(false))
        {
            appendLog(QStringLiteral("Import Job skipped: scan_manifest.json COMPLETED=false for %1").arg(projectionDir));
            ++skipped;
            continue;
        }
        const QJsonObject acq = manifest.value(QStringLiteral("ACQUISITION")).toObject();
        const bool manifestAutoCor = acq.value(QStringLiteral("AUTO_COR")).toBool(false);
        if (!job.contains(QStringLiteral("JOB_SOURCE")))
        {
            job["JOB_SOURCE"] = manifestAutoCor ? QStringLiteral("CT_ACQUISITION") : QStringLiteral("MANUAL_QUEUE");
        }
        if (job.value(QStringLiteral("JOB_SOURCE")).toString() == QStringLiteral("CT_ACQUISITION"))
        {
            QJsonObject geometry = job.value(QStringLiteral("GEOMETRY")).toObject();
            geometry["COR_AUTO"] = true;
            geometry["COR_SHIFT"] = 0.0;
            job["GEOMETRY"] = geometry;
        }

        QString outputDir = QDir::fromNativeSeparators(job.value(QStringLiteral("OUTPUT_DIR")).toString().trimmed());
        if (outputDir.isEmpty())
        {
            outputDir = QDir(projectionDir).filePath(QStringLiteral("recon"));
        }

        maxIndex += 10;
        job["TYPE"] = "CT_STREAM_RECON_JOB";
        job["VERSION"] = std::max(2, job.value(QStringLiteral("VERSION")).toInt(2));
        job["STATUS"] = "pending";
        job["QUEUE_INDEX"] = maxIndex;
        job["IMPORTED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        job["SOURCE_JOB_PATH"] = QDir::fromNativeSeparators(sourcePath);
        job["PROJECTION_DIR"] = projectionDir;
        job["OUTPUT_DIR"] = outputDir;

        QString base = queueBaseNameFromPath(sourcePath);
        if (base.isEmpty())
        {
            base = sanitizeQueueName(QFileInfo(projectionDir).fileName());
        }
        QString targetPath = pendingDir.filePath(queueFileName(maxIndex, base));
        int duplicate = 1;
        while (QFileInfo::exists(targetPath))
        {
            targetPath = pendingDir.filePath(queueFileName(maxIndex, QStringLiteral("%1_%2").arg(base).arg(duplicate++)));
        }

        if (!writeJsonFileAtomic(targetPath, job))
        {
            appendLog(QStringLiteral("Import Job failed: cannot write %1").arg(targetPath));
            ++skipped;
            continue;
        }
        appendLog(QStringLiteral("Imported recon job: %1").arg(targetPath));
        ++imported;
    }

    normalizePendingReconQueue(false);
    refreshReconQueue();
    startNextQueuedReconstruction();
    appendLog(QStringLiteral("Import Job finished: imported=%1, skipped=%2.").arg(imported).arg(skipped));
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
    if (!cameraWarmOpenComplete_)
    {
        reconStartDeferredForCameraWarmOpen_ = true;
        writeStartupTrace(QStringLiteral("startNextQueuedReconstruction deferred: camera warm open not complete"));
        return;
    }
    if (cameraWarmOpenCompletedMs_ > 0)
    {
        const qint64 elapsedMs = QDateTime::currentMSecsSinceEpoch() - cameraWarmOpenCompletedMs_;
        if (elapsedMs >= 0 && elapsedMs < kReconQueueCameraSettleMs)
        {
            reconStartDeferredForCameraWarmOpen_ = true;
            const int remainingMs = static_cast<int>(kReconQueueCameraSettleMs - elapsedMs);
            writeStartupTrace(QStringLiteral("startNextQueuedReconstruction deferred: camera settle %1 ms remaining")
                .arg(remainingMs));
            QTimer::singleShot(std::max(1, remainingMs), this, &MainWindow::startNextQueuedReconstruction);
            return;
        }
    }
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
    const QString jobSource = job.value(QStringLiteral("JOB_SOURCE")).toString();
    const bool ctAcquisitionJob = jobSource == QStringLiteral("CT_ACQUISITION");
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
    if (ctAcquisitionJob)
    {
        cfg.geometry.corAuto = true;
        cfg.geometry.corShift = 0.0f;
    }
    cfg.preprocess.flatGap = std::max(0, acq.value(QStringLiteral("FLAT_GAP")).toInt(cfg.preprocess.flatGap));
    cfg.preprocess.flatNum = std::max(1, acq.value(QStringLiteral("FLAT_NUM")).toInt(cfg.preprocess.flatNum));
    const QJsonArray manifestFlatGroups = manifest.value(QStringLiteral("FLAT_GROUPS")).toArray();
    if (!manifestFlatGroups.isEmpty())
    {
        cfg.preprocess.flatGroups.clear();
        for (const QJsonValue& value : manifestFlatGroups)
        {
            const QJsonObject obj = value.toObject();
            FlatGroupConfig group;
            group.label = obj.value(QStringLiteral("LABEL")).toString();
            group.afterProjectionCount = obj.value(QStringLiteral("AFTER_PROJECTION_COUNT")).toInt();
            group.firstFlatIndex = obj.value(QStringLiteral("FIRST_FLAT_INDEX")).toInt();
            group.flatCount = obj.value(QStringLiteral("FLAT_COUNT")).toInt();
            if (group.flatCount > 0)
            {
                cfg.preprocess.flatGroups.append(group);
            }
        }
    }
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
    const int tomoStartIndex = resolveTomoStartIndexForConfig(cfg);
    const QString firstTomo = QDir(projectionDir).filePath(numberedFileName(cfg.io.tomoPrefix,
        cfg.io.inputPadding, tomoStartIndex, cfg.io.inputExt));
    const QString lastTomo = QDir(projectionDir).filePath(numberedFileName(cfg.io.tomoPrefix,
        cfg.io.inputPadding, tomoStartIndex + cfg.geometry.totalAngles - 1, cfg.io.inputExt));
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
        QString missingFlat;
        if (!cfg.preprocess.flatGroups.isEmpty())
        {
            for (const FlatGroupConfig& group : cfg.preprocess.flatGroups)
            {
                const QString firstFlat = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
                    cfg.io.inputPadding, group.firstFlatIndex, cfg.io.inputExt));
                const QString lastFlat = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
                    cfg.io.inputPadding, group.firstFlatIndex + group.flatCount - 1, cfg.io.inputExt));
                if (!QFileInfo::exists(firstFlat))
                {
                    missingFlat = firstFlat;
                    break;
                }
                if (!QFileInfo::exists(lastFlat))
                {
                    missingFlat = lastFlat;
                    break;
                }
            }
        }
        else
        {
            int firstFlatIndex = 0;
            bool foundFlat = false;
            for (int index = 0; index < 100000; ++index)
            {
                const QString flatPath = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
                    cfg.io.inputPadding, index, cfg.io.inputExt));
                if (QFileInfo::exists(flatPath))
                {
                    firstFlatIndex = index;
                    foundFlat = true;
                    break;
                }
            }
            const QVector<FlatGroupConfig> inferredGroups = foundFlat ? inferQueueFlatGroups(cfg, firstFlatIndex) : QVector<FlatGroupConfig>();
            if (!foundFlat)
            {
                missingFlat = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
                    cfg.io.inputPadding, 0, cfg.io.inputExt));
            }
            for (const FlatGroupConfig& group : inferredGroups)
            {
                const QString firstFlat = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
                    cfg.io.inputPadding, group.firstFlatIndex, cfg.io.inputExt));
                const QString lastFlat = QDir(projectionDir).filePath(numberedFileName(cfg.io.flatPrefix,
                    cfg.io.inputPadding, group.firstFlatIndex + group.flatCount - 1, cfg.io.inputExt));
                if (!QFileInfo::exists(firstFlat))
                {
                    missingFlat = firstFlat;
                    break;
                }
                if (!QFileInfo::exists(lastFlat))
                {
                    missingFlat = lastFlat;
                    break;
                }
            }
        }
        if (!missingFlat.isEmpty())
        {
            const QString message = QStringLiteral("Flat correction requested but flat file is missing: %1").arg(missingFlat);
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
    writeStartupTrace(QStringLiteral("startQueuedReconstruction applyConfig begin: %1").arg(runningPath));
    applyConfig(cfg);
    writeStartupTrace(QStringLiteral("startQueuedReconstruction applyConfig end: %1").arg(runningPath));
    appendLog(QStringLiteral("Starting queued reconstruction: %1").arg(runningPath));
    writeStartupTrace(QStringLiteral("startQueuedReconstruction refreshReconQueue begin: %1").arg(runningPath));
    refreshReconQueue();
    writeStartupTrace(QStringLiteral("startQueuedReconstruction refreshReconQueue end: %1").arg(runningPath));
    writeStartupTrace(QStringLiteral("startQueuedReconstruction controller start begin: %1").arg(runningPath));
    controller_.startReconstruction(cfg);
    writeStartupTrace(QStringLiteral("startQueuedReconstruction controller start end: %1").arg(runningPath));
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
    const bool autoCorMode = job.value(QStringLiteral("GEOMETRY")).toObject().value(QStringLiteral("COR_AUTO")).toBool(hasAutoCor);
    result["COR_MODE"] = autoCorMode ? QStringLiteral("auto") : QStringLiteral("manual");
    if (hasAutoCor)
    {
        result["AUTO_COR_SHIFT"] = autoCor;
        result["COR_SHIFT_COMPUTED"] = autoCor;
    }
    result["FINAL_COR_SHIFT"] = finalCor;
    result["COR_SHIFT_USED"] = finalCor;
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
    stopButton_->setEnabled(false);
    stopButton_->setText(QStringLiteral("Stopping..."));
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
        batchSpin_->setEnabled(true);
        batchSpin_->setToolTip(offline
            ? QStringLiteral("Projection batch size for offline TIFF read/decode and GPU preprocessing before full-angle z-chunk reconstruction.")
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
        phaseEnableCheck_->setEnabled(true);
        phaseEnableCheck_->setToolTip(QStringLiteral("Apply Paganin phase retrieval on the GPU projection batch after flat/dark correction and before log/taper."));
    }

    if (energySpin_ != nullptr) { energySpin_->setEnabled(true); }
    if (distanceSpin_ != nullptr) { distanceSpin_->setEnabled(true); }
    if (pixelSpin_ != nullptr) { pixelSpin_->setEnabled(true); }
    if (alphaSpin_ != nullptr) { alphaSpin_->setEnabled(true); }

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
    for (auto it = motorAxes_.begin(); it != motorAxes_.end(); ++it)
    {
        kohzuController_->addAxis(it->axis);
    }
    kohzuController_->moveToThread(&kohzuThread_);
    connect(kohzuController_, &KohzuController::connectionChanged, this, [this](bool connected) {
        if (shuttingDown_) { return; }
        motorConnected_ = connected;
        motorConnectionLabel_->setText(connected ? QStringLiteral("Connected") : QStringLiteral("Disconnected"));
        updateMotorUiEnabled();
        if (connected && ctRecoveryPending_ && !ctRecoveryConfirmed_)
        {
            appendLog(QStringLiteral("CT recovery: controller connected; checking R/Y idle state."));
            QMetaObject::invokeMethod(kohzuController_, "recoverAfterInterruptedSession", Qt::QueuedConnection);
        }
    });
    connect(kohzuController_, &KohzuController::axisUpdated, this, &MainWindow::updateMotorAxisPosition);
    connect(kohzuController_, &KohzuController::axisActiveChanged, this, &MainWindow::updateMotorAxisActive);
    connect(kohzuController_, &KohzuController::logMessage, this, [this](const QString& text) {
        if (isKohzuErrorLog(text))
        {
            appendLog(text);
        }
    });
    connect(kohzuController_, &KohzuController::commandFinished, this, [this](const QString& label, bool ok, const QString& response) {
        if (ctHardwareStopPending_ && label == QStringLiteral("Kohzu stop all") && ok)
        {
            ctHardwareStopPending_ = false;
            writeCtRecoveryState(QStringLiteral("STOPPED"), QStringLiteral("Kohzu controlled stop acknowledged."));
        }
        if (ctRecoveryPending_ && label == QStringLiteral("Kohzu recovery") && ok)
        {
            ctRecoveryConfirmed_ = true;
            ctRecoveryPending_ = false;
            writeCtRecoveryState(QStringLiteral("RECOVERED"), QStringLiteral("Controlled stop and idle polling completed."));
            appendLog(QStringLiteral("CT recovery confirmed; new acquisitions are enabled."));
        }
        else if (ctRecoveryPending_ && label == QStringLiteral("Kohzu recovery") && !ok)
        {
            appendLog(QStringLiteral("CT recovery remains blocked: %1").arg(response));
        }
    });
    connect(kohzuController_, &KohzuController::commandFinished, this, [this](const QString& label, bool ok, const QString& response) {
        if (shuttingDown_) { return; }
        if (!ok)
        {
            appendLog(QStringLiteral("%1 failed: %2").arg(label, response));
        }
        if (ctRunning_ && label.startsWith(QStringLiteral("CT speed table ")))
        {
            if (!ok)
            {
                finishCtAcquisition(false, QStringLiteral("%1 failed: %2").arg(label, response));
                return;
            }
            sendNextCtMotorSpeedTableSetup();
            return;
        }
        if (ctRunning_ && ok && label.startsWith(QStringLiteral("CT Fly R scan to")) && cameraWorker_)
        {
            QMetaObject::invokeMethod(cameraWorker_, "armContinuousFrameGate", Qt::DirectConnection);
            appendLog(QStringLiteral("CT fly device-frame gate armed after R sweep command acceptance."));
        }
        if (ctRunning_ && !ok && label.startsWith(QStringLiteral("CT ")))
        {
            finishCtAcquisition(false, QStringLiteral("%1 failed: %2").arg(label, response));
        }
    });
    kohzuThread_.start();
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
        finishCtMotorTiming(action);
        if (action == QStringLiteral("flat_out"))
        {
            ctYIsOutForFlat_ = true;
        }
        else if (action == QStringLiteral("flat_return"))
        {
            ctYIsOutForFlat_ = false;
        }
        if (ctSettings_.mode == CtAcquisitionMode::FlyScan)
        {
            if (action == QStringLiteral("r_fly_start"))
            {
                QTimer::singleShot(0, this, &MainWindow::startCtFlyScanMove);
            }
            else if (action == QStringLiteral("r_fly_scan"))
            {
                ctFlyScanMoveDone_ = true;
                appendLog(QStringLiteral("CT fly R scan move completed."));
                if (!ctFlyScanCapturing_)
                {
                    QTimer::singleShot(0, this, &MainWindow::advanceCtFlyAcquisition);
                }
            }
            else
            {
                QTimer::singleShot(0, this, &MainWindow::advanceCtFlyAcquisition);
            }
        }
        else
        {
            QTimer::singleShot(0, this, &MainWindow::advanceCtAcquisition);
        }
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

    const QString label = QStringLiteral("%1 -> %2 %3")
        .arg(name)
        .arg(target, 0, 'f', 4)
        .arg(it->unit);
    const MotorSpeedTableParams speedParams = makeMotorSpeedTableParams(
        name,
        kMotorManualSpeedTable,
        it->pulsePerUnit,
        it->unit,
        it->baseSpeedSpin ? it->baseSpeedSpin->value() : 1.0,
        it->speedSpin ? it->speedSpin->value() : 2.0,
        it->accelSpin ? it->accelSpin->value() : 0.5);
    it->active = true;
    it->goButton->setEnabled(false);
    QMetaObject::invokeMethod(
        kohzuController_,
        "configureSpeedTableAndMoveAbsolute",
        Qt::QueuedConnection,
        Q_ARG(int, it->axis),
        Q_ARG(qint64, pulse),
        Q_ARG(int, speedParams.speedTable),
        Q_ARG(int, speedParams.startSpeedPps),
        Q_ARG(int, speedParams.topSpeedPps),
        Q_ARG(int, speedParams.accelTime10ms),
        Q_ARG(int, speedParams.decelTime10ms),
        Q_ARG(int, speedParams.accelPattern),
        Q_ARG(QString, speedParams.summary),
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

    const QString label = QStringLiteral("%1 %2%3 %4")
        .arg(name)
        .arg(relative >= 0.0 ? QStringLiteral("+") : QStringLiteral(""))
        .arg(relative, 0, 'f', 4)
        .arg(it->unit);
    const MotorSpeedTableParams speedParams = makeMotorSpeedTableParams(
        name,
        kMotorManualSpeedTable,
        it->pulsePerUnit,
        it->unit,
        it->baseSpeedSpin ? it->baseSpeedSpin->value() : 1.0,
        it->speedSpin ? it->speedSpin->value() : 2.0,
        it->accelSpin ? it->accelSpin->value() : 0.5);
    it->active = true;
    it->relativeMinusButton->setEnabled(false);
    it->relativePlusButton->setEnabled(false);
    it->goButton->setEnabled(false);
    QMetaObject::invokeMethod(
        kohzuController_,
        "configureSpeedTableAndMoveRelative",
        Qt::QueuedConnection,
        Q_ARG(int, it->axis),
        Q_ARG(qint64, pulse),
        Q_ARG(int, speedParams.speedTable),
        Q_ARG(int, speedParams.startSpeedPps),
        Q_ARG(int, speedParams.topSpeedPps),
        Q_ARG(int, speedParams.accelTime10ms),
        Q_ARG(int, speedParams.decelTime10ms),
        Q_ARG(int, speedParams.accelPattern),
        Q_ARG(QString, speedParams.summary),
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
        if (it->speedWidget)
        {
            it->speedWidget->setVisible(motorSpeedMode_);
        }
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
        if (it->baseSpeedSpin)
        {
            it->baseSpeedSpin->setEnabled(motorSpeedMode_);
        }
        if (it->speedSpin)
        {
            it->speedSpin->setEnabled(motorSpeedMode_);
        }
        if (it->accelSpin)
        {
            it->accelSpin->setEnabled(motorSpeedMode_);
        }
    }
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    writeStartupTrace(QStringLiteral("closeEvent enter"));
    saveSessionConfig();
    shutdownWorkers();
    writeStartupTrace(QStringLiteral("closeEvent accept"));
    event->accept();
}

void MainWindow::shutdownWorkers()
{
    if (shuttingDown_)
    {
        return;
    }
    shuttingDown_ = true;

    writeStartupTrace(QStringLiteral("shutdownWorkers begin"));
    controller_.shutdown();
    writeStartupTrace(QStringLiteral("shutdownWorkers controller shutdown done"));

    if (ctReplayWorker_)
    {
        writeStartupTrace(QStringLiteral("shutdownWorkers ctReplay begin"));
        disconnect(ctReplayWorker_, nullptr, this, nullptr);
        QMetaObject::invokeMethod(ctReplayWorker_, "stopReplay", Qt::QueuedConnection);
        ctReplayThread_.quit();
        ctReplayThread_.wait(3000);
        ctReplayWorker_ = nullptr;
        writeStartupTrace(QStringLiteral("shutdownWorkers ctReplay done"));
    }
    else
    {
        ctReplayThread_.quit();
        ctReplayThread_.wait(3000);
    }

    if (ctWriter_)
    {
        writeStartupTrace(QStringLiteral("shutdownWorkers ctWriter begin"));
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
        writeStartupTrace(QStringLiteral("shutdownWorkers ctWriter done"));
    }
    else
    {
        ctWriterThread_.quit();
        ctWriterThread_.wait(3000);
    }

    if (cameraWorker_)
    {
        writeStartupTrace(QStringLiteral("shutdownWorkers camera begin"));
        QMetaObject::invokeMethod(cameraWorker_, "stopCapture", Qt::DirectConnection);
        disconnect(cameraWorker_, nullptr, this, nullptr);
        PvcamCameraWorker* worker = cameraWorker_;
        QThread* guiThread = thread();
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
        writeStartupTrace(QStringLiteral("shutdownWorkers camera done"));
    }
    else
    {
        cameraThread_.quit();
        cameraThread_.wait(7000);
    }

    if (kohzuController_)
    {
        writeStartupTrace(QStringLiteral("shutdownWorkers kohzu begin"));
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
        writeStartupTrace(QStringLiteral("shutdownWorkers kohzu done"));
    }
    else
    {
        kohzuThread_.quit();
        kohzuThread_.wait(1500);
    }
    writeStartupTrace(QStringLiteral("shutdownWorkers end"));
}

} // namespace ct
