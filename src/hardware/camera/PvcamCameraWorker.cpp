#include "hardware/camera/PvcamCameraWorker.h"
#include "core/acq/PvcamRingDiagnostics.h"

#include "core/io/ImageFileIo.h"

#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QCoreApplication>
#include <QMetaType>
#include <QStringList>
#include <QTextStream>
#include <QThread>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef CT_HAS_PVCAM
#include <master.h>
#include <pvcam.h>
#endif

namespace ct
{

namespace
{

void writePvcamTrace(const QString& text)
{
    QFile file(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("ct_startup_trace.log")));
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        QTextStream stream(&file);
        stream << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << " " << text << '\n';
    }
}

#ifdef CT_HAS_PVCAM

QString pvcamError()
{
    char message[ERROR_MSG_LEN] = {};
    const int16 code = pl_error_code();
    pl_error_message(code, message);
    return QStringLiteral("PVCAM error %1: %2").arg(code).arg(QString::fromLatin1(message));
}

template <typename T>
bool getParam(int16 hcam, uns32 id, int16 attr, T& value)
{
    return PV_OK == pl_get_param(hcam, id, attr, &value);
}

bool enumContains(int16 hcam, uns32 paramId, int32 wanted)
{
    uns32 count = 0;
    if (PV_OK != pl_get_param(hcam, paramId, ATTR_COUNT, &count))
    {
        return false;
    }

    for (uns32 index = 0; index < count; ++index)
    {
        int32 value = 0;
        uns32 length = 0;
        if (PV_OK != pl_enum_str_length(hcam, paramId, index, &length) || length == 0)
        {
            continue;
        }

        std::vector<char> text(length + 1, '\0');
        if (PV_OK == pl_get_enum_param(hcam, paramId, index, &value, text.data(), length)
            && value == wanted)
        {
            return true;
        }
    }
    return false;
}

bool paramAvailable(int16 hcam, uns32 paramId)
{
    rs_bool available = FALSE;
    return PV_OK == pl_get_param(hcam, paramId, ATTR_AVAIL, &available) && available == TRUE;
}

bool paramWritable(int16 hcam, uns32 paramId, QString* error)
{
    if (!paramAvailable(hcam, paramId))
    {
        if (error != nullptr)
        {
            *error = QStringLiteral("PVCAM parameter %1 is not available.").arg(paramId);
        }
        return false;
    }

    uns16 access = 0;
    if (PV_OK != pl_get_param(hcam, paramId, ATTR_ACCESS, &access))
    {
        if (error != nullptr)
        {
            *error = pvcamError();
        }
        return false;
    }

    if (access != ACC_READ_WRITE && access != ACC_WRITE_ONLY)
    {
        if (error != nullptr)
        {
            *error = QStringLiteral("PVCAM parameter %1 is not writable (access=%2).").arg(paramId).arg(access);
        }
        return false;
    }
    return true;
}

template <typename T>
bool setParamIfAvailable(int16 hcam, uns32 paramId, const T& value, QString* error)
{
    if (!paramWritable(hcam, paramId, error))
    {
        return false;
    }

    T mutableValue = value;
    if (PV_OK != pl_set_param(hcam, paramId, &mutableValue))
    {
        if (error != nullptr)
        {
            *error = pvcamError();
        }
        return false;
    }
    return true;
}

bool firstEnumValue(int16 hcam, uns32 paramId, int32& value, QString& name)
{
    uns32 count = 0;
    if (PV_OK != pl_get_param(hcam, paramId, ATTR_COUNT, &count) || count == 0)
    {
        return false;
    }

    uns32 length = 0;
    if (PV_OK != pl_enum_str_length(hcam, paramId, 0, &length) || length == 0)
    {
        return false;
    }

    std::vector<char> text(length + 1, '\0');
    if (PV_OK != pl_get_enum_param(hcam, paramId, 0, &value, text.data(), length))
    {
        return false;
    }
    name = QString::fromLatin1(text.data());
    return true;
}

struct EnumValue
{
    int32 value = 0;
    QString name;
};

std::vector<EnumValue> enumValues(int16 hcam, uns32 paramId)
{
    std::vector<EnumValue> values;
    uns32 count = 0;
    if (PV_OK != pl_get_param(hcam, paramId, ATTR_COUNT, &count))
    {
        return values;
    }

    for (uns32 index = 0; index < count; ++index)
    {
        uns32 length = 0;
        if (PV_OK != pl_enum_str_length(hcam, paramId, index, &length) || length == 0)
        {
            continue;
        }

        std::vector<char> text(length + 1, '\0');
        int32 value = 0;
        if (PV_OK == pl_get_enum_param(hcam, paramId, index, &value, text.data(), length))
        {
            values.push_back({ value, QString::fromLatin1(text.data()) });
        }
    }
    return values;
}

QString currentEnumValueName(int16 hcam, uns32 paramId)
{
    int32 currentValue = 0;
    if (!getParam(hcam, paramId, ATTR_CURRENT, currentValue))
    {
        return {};
    }
    for (const EnumValue& value : enumValues(hcam, paramId))
    {
        if (value.value == currentValue)
        {
            return value.name;
        }
    }
    return QString::number(currentValue);
}

QString readStringParam(int16 hcam, uns32 paramId, int maxLength)
{
    std::vector<char> text(static_cast<size_t>(std::max(1, maxLength)), '\0');
    if (PV_OK == pl_get_param(hcam, paramId, ATTR_CURRENT, text.data()))
    {
        return QString::fromLatin1(text.data());
    }
    return {};
}

std::vector<int16> int16IndexRange(int16 hcam, uns32 paramId, int16 fallbackMin, int16 fallbackMax)
{
    int16 minValue = fallbackMin;
    int16 maxValue = fallbackMax;
    getParam(hcam, paramId, ATTR_MIN, minValue);
    getParam(hcam, paramId, ATTR_MAX, maxValue);
    if (maxValue < minValue)
    {
        return {};
    }

    minValue = static_cast<int16>(std::max<int>(-1, minValue));
    maxValue = static_cast<int16>(std::min<int>(64, maxValue));
    std::vector<int16> values;
    for (int value = minValue; value <= maxValue; ++value)
    {
        values.push_back(static_cast<int16>(value));
    }
    return values;
}

int16 currentCameraBitDepth(int16 hcam)
{
    int16 bitDepth = 0;
    if (!getParam(hcam, PARAM_BIT_DEPTH_HOST, ATTR_CURRENT, bitDepth))
    {
        getParam(hcam, PARAM_BIT_DEPTH, ATTR_CURRENT, bitDepth);
    }
    return bitDepth;
}

struct KinetixReadoutCandidate
{
    int32 readoutPort = 0;
    QString readoutPortName;
    int16 speedIndex = 0;
    QString speedName;
    int16 gainIndex = 0;
    QString gainName;
    int16 bitDepth = 0;
    int score = std::numeric_limits<int>::min();
};

int scoreKinetixDynamicRangeCandidate(const KinetixReadoutCandidate& candidate)
{
    const QString text = QStringLiteral("%1 %2 %3")
        .arg(candidate.readoutPortName, candidate.speedName, candidate.gainName)
        .toLower();
    int score = 0;
    score += candidate.bitDepth >= 16 ? 1000 : -1000;
    if (text.contains(QStringLiteral("dynamic")) && text.contains(QStringLiteral("range"))) { score += 600; }
    if (text.contains(QStringLiteral("dynamic"))) { score += 250; }
    if (text.contains(QStringLiteral("range"))) { score += 150; }
    if (text.contains(QStringLiteral("16"))) { score += 150; }
    if (text.contains(QStringLiteral("hdr"))) { score += 150; }
    if (text.contains(QStringLiteral("sub")) || text.contains(QStringLiteral("electron"))) { score -= 400; }
    if (text.contains(QStringLiteral("sensitivity"))) { score -= 150; }
    if (text.contains(QStringLiteral("speed"))) { score -= 150; }
    if (text.contains(QStringLiteral("12"))) { score -= 150; }
    if (text.contains(QStringLiteral("8"))) { score -= 250; }
    return score;
}

bool selectKinetixDynamicRangeReadout(int16 hcam, QStringList& messages)
{
    std::vector<EnumValue> readoutPorts;
    if (paramAvailable(hcam, PARAM_READOUT_PORT))
    {
        readoutPorts = enumValues(hcam, PARAM_READOUT_PORT);
    }
    if (readoutPorts.empty())
    {
        int32 currentPort = 0;
        getParam(hcam, PARAM_READOUT_PORT, ATTR_CURRENT, currentPort);
        readoutPorts.push_back({ currentPort, QStringLiteral("current") });
    }

    KinetixReadoutCandidate best;
    QStringList tried;
    for (const EnumValue& port : readoutPorts)
    {
        if (paramAvailable(hcam, PARAM_READOUT_PORT))
        {
            int32 readoutPort = port.value;
            if (PV_OK != pl_set_param(hcam, PARAM_READOUT_PORT, &readoutPort))
            {
                tried << QStringLiteral("readout '%1' failed: %2").arg(port.name, pvcamError());
                continue;
            }
        }

        std::vector<int16> speedIndices = paramAvailable(hcam, PARAM_SPDTAB_INDEX)
            ? int16IndexRange(hcam, PARAM_SPDTAB_INDEX, 0, 0)
            : std::vector<int16>{ 0 };
        for (int16 speedIndex : speedIndices)
        {
            if (paramAvailable(hcam, PARAM_SPDTAB_INDEX))
            {
                int16 mutableSpeedIndex = speedIndex;
                if (PV_OK != pl_set_param(hcam, PARAM_SPDTAB_INDEX, &mutableSpeedIndex))
                {
                    continue;
                }
                getParam(hcam, PARAM_SPDTAB_INDEX, ATTR_CURRENT, speedIndex);
            }
            const QString speedName = readStringParam(hcam, PARAM_SPDTAB_NAME, MAX_SPDTAB_NAME_LEN);

            std::vector<int16> gainIndices = paramAvailable(hcam, PARAM_GAIN_INDEX)
                ? int16IndexRange(hcam, PARAM_GAIN_INDEX, 1, 1)
                : std::vector<int16>{ 0 };
            for (int16 gainIndex : gainIndices)
            {
                if (paramAvailable(hcam, PARAM_GAIN_INDEX))
                {
                    int16 mutableGainIndex = gainIndex;
                    if (PV_OK != pl_set_param(hcam, PARAM_GAIN_INDEX, &mutableGainIndex))
                    {
                        continue;
                    }
                    getParam(hcam, PARAM_GAIN_INDEX, ATTR_CURRENT, gainIndex);
                }
                const QString gainName = readStringParam(hcam, PARAM_GAIN_NAME, MAX_GAIN_NAME_LEN);
                KinetixReadoutCandidate candidate;
                candidate.readoutPort = port.value;
                candidate.readoutPortName = port.name;
                candidate.speedIndex = speedIndex;
                candidate.speedName = speedName;
                candidate.gainIndex = gainIndex;
                candidate.gainName = gainName;
                candidate.bitDepth = currentCameraBitDepth(hcam);
                candidate.score = scoreKinetixDynamicRangeCandidate(candidate);
                tried << QStringLiteral("port='%1' speed=%2 '%3' gain=%4 '%5' bitDepth=%6 score=%7")
                    .arg(candidate.readoutPortName)
                    .arg(candidate.speedIndex)
                    .arg(candidate.speedName)
                    .arg(candidate.gainIndex)
                    .arg(candidate.gainName)
                    .arg(candidate.bitDepth)
                    .arg(candidate.score);
                if (candidate.score > best.score)
                {
                    best = candidate;
                }
            }
        }
    }

    if (best.score <= std::numeric_limits<int>::min() / 2)
    {
        messages << QStringLiteral("16-bit Dynamic Range search found no readout candidates");
        return false;
    }

    if (paramAvailable(hcam, PARAM_READOUT_PORT))
    {
        int32 readoutPort = best.readoutPort;
        if (PV_OK != pl_set_param(hcam, PARAM_READOUT_PORT, &readoutPort))
        {
            messages << QStringLiteral("16-bit Dynamic Range readout port apply failed: %1").arg(pvcamError());
            return false;
        }
    }
    if (paramAvailable(hcam, PARAM_SPDTAB_INDEX))
    {
        int16 speedIndex = best.speedIndex;
        if (PV_OK != pl_set_param(hcam, PARAM_SPDTAB_INDEX, &speedIndex))
        {
            messages << QStringLiteral("16-bit Dynamic Range speed apply failed: %1").arg(pvcamError());
            return false;
        }
    }
    if (paramAvailable(hcam, PARAM_GAIN_INDEX))
    {
        int16 gainIndex = best.gainIndex;
        if (PV_OK != pl_set_param(hcam, PARAM_GAIN_INDEX, &gainIndex))
        {
            messages << QStringLiteral("16-bit Dynamic Range gain apply failed: %1").arg(pvcamError());
            return false;
        }
    }

    const int16 finalBitDepth = currentCameraBitDepth(hcam);
    messages << QStringLiteral("preferred 16-bit Dynamic Range: readout '%1', speed index %2 '%3', gain index %4 '%5', bitDepth %6")
        .arg(best.readoutPortName)
        .arg(best.speedIndex)
        .arg(best.speedName)
        .arg(best.gainIndex)
        .arg(best.gainName)
        .arg(finalBitDepth);
    if (finalBitDepth < 16)
    {
        messages << QStringLiteral("warning: 16-bit Dynamic Range selection did not report 16-bit output; candidates: %1")
            .arg(tried.join(QStringLiteral("; ")));
    }
    return finalBitDepth >= 16;
}

QString initializeKinetixReadout(int16 hcam)
{
    QStringList messages;

    pl_pp_reset(hcam);

    if (!selectKinetixDynamicRangeReadout(hcam, messages))
    {
        int32 readoutPort = 0;
        QString readoutPortName;
        if (paramAvailable(hcam, PARAM_READOUT_PORT)
            && firstEnumValue(hcam, PARAM_READOUT_PORT, readoutPort, readoutPortName))
        {
            if (PV_OK != pl_set_param(hcam, PARAM_READOUT_PORT, &readoutPort))
            {
                throw std::runtime_error(("Failed to set PVCAM readout port: " + pvcamError()).toStdString());
            }
            messages << QStringLiteral("fallback readout port '%1'").arg(readoutPortName);
        }

        if (paramAvailable(hcam, PARAM_SPDTAB_INDEX))
        {
            int16 speedIndex = 0;
            if (PV_OK != pl_set_param(hcam, PARAM_SPDTAB_INDEX, &speedIndex))
            {
                throw std::runtime_error(("Failed to set PVCAM speed index: " + pvcamError()).toStdString());
            }
            const QString speedName = readStringParam(hcam, PARAM_SPDTAB_NAME, MAX_SPDTAB_NAME_LEN);
            messages << QStringLiteral("fallback speed index %1%2")
                .arg(speedIndex)
                .arg(speedName.isEmpty() ? QString() : QStringLiteral(" '%1'").arg(speedName));
        }

        if (paramAvailable(hcam, PARAM_GAIN_INDEX))
        {
            int16 gainIndex = 1;
            if (PV_OK != pl_set_param(hcam, PARAM_GAIN_INDEX, &gainIndex))
            {
                throw std::runtime_error(("Failed to set PVCAM gain index: " + pvcamError()).toStdString());
            }
            const QString gainName = readStringParam(hcam, PARAM_GAIN_NAME, MAX_GAIN_NAME_LEN);
            messages << QStringLiteral("fallback gain index %1%2")
                .arg(gainIndex)
                .arg(gainName.isEmpty() ? QString() : QStringLiteral(" '%1'").arg(gainName));
        }
    }

    if (paramAvailable(hcam, PARAM_CLEAR_CYCLES))
    {
        uns16 clearCycles = 2;
        pl_set_param(hcam, PARAM_CLEAR_CYCLES, &clearCycles);
    }

    if (paramAvailable(hcam, PARAM_PMODE))
    {
        rs_bool frameCapable = FALSE;
        int32 pMode = PMODE_NORMAL;
        if (paramAvailable(hcam, PARAM_FRAME_CAPABLE)
            && PV_OK == pl_get_param(hcam, PARAM_FRAME_CAPABLE, ATTR_CURRENT, &frameCapable)
            && frameCapable == TRUE)
        {
            pMode = PMODE_FT;
        }
        if (PV_OK != pl_set_param(hcam, PARAM_PMODE, &pMode))
        {
            throw std::runtime_error(("Failed to set PVCAM PMODE: " + pvcamError()).toStdString());
        }
        messages << QStringLiteral("PMODE %1").arg(pMode == PMODE_FT ? QStringLiteral("FT") : QStringLiteral("NORMAL"));
    }

    return messages.join(QStringLiteral(", "));
}

QString exposureModeLabel(int16 expMode)
{
    if (expMode == TIMED_MODE)
    {
        return QStringLiteral("TIMED_MODE");
    }
    if ((expMode & EXT_TRIG_INTERNAL) == EXT_TRIG_INTERNAL)
    {
        return QStringLiteral("EXT_TRIG_INTERNAL | exposeOut %1").arg(expMode & ~EXT_TRIG_INTERNAL);
    }
    return QStringLiteral("mode %1").arg(expMode);
}

QString exposureResolutionLabel(int32 expRes)
{
    switch (expRes)
    {
    case EXP_RES_ONE_MILLISEC:
        return QStringLiteral("milliseconds");
    case EXP_RES_ONE_MICROSEC:
        return QStringLiteral("microseconds");
    case EXP_RES_ONE_SEC:
        return QStringLiteral("seconds");
    default:
        return QStringLiteral("unknown(%1)").arg(expRes);
    }
}

struct ExposureResolution
{
    int32 mode = EXP_RES_ONE_MILLISEC;
    double unitsPerMillisecond = 1.0;
    QString logText;
};

int32 pvcamResolutionForUnit(PvcamExposureUnit unit)
{
    switch (unit)
    {
    case PvcamExposureUnit::Microseconds:
        return EXP_RES_ONE_MICROSEC;
    case PvcamExposureUnit::Seconds:
        return EXP_RES_ONE_SEC;
    case PvcamExposureUnit::Milliseconds:
    default:
        return EXP_RES_ONE_MILLISEC;
    }
}

QString exposureUnitLabel(PvcamExposureUnit unit)
{
    switch (unit)
    {
    case PvcamExposureUnit::Microseconds:
        return QStringLiteral("us");
    case PvcamExposureUnit::Seconds:
        return QStringLiteral("s");
    case PvcamExposureUnit::Milliseconds:
    default:
        return QStringLiteral("ms");
    }
}

double exposureMilliseconds(double value, PvcamExposureUnit unit)
{
    switch (unit)
    {
    case PvcamExposureUnit::Microseconds:
        return value / 1000.0;
    case PvcamExposureUnit::Seconds:
        return value * 1000.0;
    case PvcamExposureUnit::Milliseconds:
    default:
        return value;
    }
}

ExposureResolution configureExposureResolution(int16 hcam, PvcamExposureUnit requestedUnit)
{
    ExposureResolution result;

    int32 current = EXP_RES_ONE_MILLISEC;
    const bool hasCurrent = getParam(hcam, PARAM_EXP_RES, ATTR_CURRENT, current);
    const int32 requestedResolution = pvcamResolutionForUnit(requestedUnit);

    uns32 count = 0;
    if (PV_OK == pl_get_param(hcam, PARAM_EXP_RES, ATTR_COUNT, &count))
    {
        int preferredIndex = -1;
        int millisecondIndex = -1;
        for (uns32 index = 0; index < count; ++index)
        {
            int32 value = 0;
            uns32 length = 0;
            if (PV_OK != pl_enum_str_length(hcam, PARAM_EXP_RES, index, &length) || length == 0)
            {
                continue;
            }

            std::vector<char> text(length + 1, '\0');
            if (PV_OK == pl_get_enum_param(hcam, PARAM_EXP_RES, index, &value, text.data(), length))
            {
                if (value == requestedResolution)
                {
                    preferredIndex = static_cast<int>(index);
                }
                if (value == EXP_RES_ONE_MILLISEC)
                {
                    millisecondIndex = static_cast<int>(index);
                }
            }
        }

        const int selectedIndex = preferredIndex >= 0 ? preferredIndex : millisecondIndex;
        if (selectedIndex >= 0)
        {
            uns16 expResIndex = static_cast<uns16>(selectedIndex);
            if (PV_OK == pl_set_param(hcam, PARAM_EXP_RES_INDEX, &expResIndex))
            {
                current = preferredIndex >= 0 ? requestedResolution : EXP_RES_ONE_MILLISEC;
                result.logText = preferredIndex >= 0
                    ? QStringLiteral("Exposure resolution set to %1.").arg(exposureResolutionLabel(current))
                    : QStringLiteral("Requested %1 exposure units unavailable; using milliseconds.")
                          .arg(exposureResolutionLabel(requestedResolution));
            }
        }
    }

    if (result.logText.isEmpty() && hasCurrent)
    {
        result.logText = QStringLiteral("Using existing exposure resolution: %1.")
            .arg(exposureResolutionLabel(current));
    }
    else if (result.logText.isEmpty())
    {
        result.logText = QStringLiteral("Exposure resolution unavailable; assuming milliseconds.");
    }

    result.mode = current;
    switch (current)
    {
    case EXP_RES_ONE_MICROSEC:
        result.unitsPerMillisecond = 1000.0;
        break;
    case EXP_RES_ONE_SEC:
        result.unitsPerMillisecond = 0.001;
        break;
    case EXP_RES_ONE_MILLISEC:
    default:
        result.unitsPerMillisecond = 1.0;
        break;
    }
    return result;
}

int16 selectInternalExposureMode(int16 hcam)
{
    if (enumContains(hcam, PARAM_EXPOSURE_MODE, TIMED_MODE))
    {
        return TIMED_MODE;
    }

    if (enumContains(hcam, PARAM_EXPOSURE_MODE, EXT_TRIG_INTERNAL))
    {
        int16 exposeOut = EXPOSE_OUT_FIRST_ROW;
        uns32 exposeOutCount = 0;
        if (PV_OK == pl_get_param(hcam, PARAM_EXPOSE_OUT_MODE, ATTR_COUNT, &exposeOutCount)
            && exposeOutCount > 0)
        {
            int32 value = 0;
            uns32 length = 0;
            if (PV_OK == pl_enum_str_length(hcam, PARAM_EXPOSE_OUT_MODE, 0, &length) && length > 0)
            {
                std::vector<char> text(length + 1, '\0');
                if (PV_OK == pl_get_enum_param(hcam, PARAM_EXPOSE_OUT_MODE, 0, &value, text.data(), length))
                {
                    exposeOut = static_cast<int16>(value);
                }
            }
        }
        return static_cast<int16>(EXT_TRIG_INTERNAL | exposeOut);
    }

    throw std::runtime_error("Camera does not report an internal timed exposure mode.");
}

rgn_type makeExplicitRoi(uns16 sensorWidth, uns16 sensorHeight, const PvcamCaptureSettings& settings)
{
    const int maxX = std::max(0, static_cast<int>(sensorWidth) - 1);
    const int maxY = std::max(0, static_cast<int>(sensorHeight) - 1);
    const int x0 = std::clamp(settings.roiX, 0, maxX);
    const int y0 = std::clamp(settings.roiY, 0, maxY);
    const int roiWidth = std::clamp(settings.roiWidth, 1, static_cast<int>(sensorWidth) - x0);
    const int roiHeight = std::clamp(settings.roiHeight, 1, static_cast<int>(sensorHeight) - y0);

    rgn_type roi {};
    roi.s1 = static_cast<uns16>(x0);
    roi.s2 = static_cast<uns16>(x0 + roiWidth - 1);
    roi.sbin = 1;
    roi.p1 = static_cast<uns16>(y0);
    roi.p2 = static_cast<uns16>(y0 + roiHeight - 1);
    roi.pbin = 1;
    return roi;
}

int roiWidth(const rgn_type& roi)
{
    return (roi.s2 - roi.s1 + 1) / roi.sbin;
}

int roiHeight(const rgn_type& roi)
{
    return (roi.p2 - roi.p1 + 1) / roi.pbin;
}

int cvTypeForImageFormat(int32 imageFormat)
{
    switch (imageFormat)
    {
    case PL_IMAGE_FORMAT_MONO8:
    case PL_IMAGE_FORMAT_BAYER8:
        return CV_8UC1;
    case PL_IMAGE_FORMAT_MONO16:
    case PL_IMAGE_FORMAT_BAYER16:
        return CV_16UC1;
    case PL_IMAGE_FORMAT_MONO32:
    case PL_IMAGE_FORMAT_BAYER32:
        return CV_32SC1;
    default:
        throw std::runtime_error("Unsupported PVCAM image format: " + std::to_string(imageFormat));
    }
}

int bytesPerPixelForCvType(int cvType)
{
    return static_cast<int>(CV_ELEM_SIZE(cvType));
}

bool waitForReadout(int16 hcam, uns32 timeoutMs, std::atomic_bool& stopRequested, uns32& bytesArrived)
{
    const qint64 deadline = QDateTime::currentMSecsSinceEpoch() + timeoutMs;
    int16 status = 0;
    int16 lastStatus = -1;
    bytesArrived = 0;

    while (QDateTime::currentMSecsSinceEpoch() < deadline)
    {
        if (stopRequested.load())
        {
            return false;
        }

        if (PV_OK != pl_exp_check_status(hcam, &status, &bytesArrived))
        {
            qWarning("pl_exp_check_status failed: %s", qPrintable(pvcamError()));
            return false;
        }

        if (status == READOUT_COMPLETE || status == FRAME_AVAILABLE)
        {
            return true;
        }

        if (status != EXPOSURE_IN_PROGRESS && status != READOUT_IN_PROGRESS)
        {
            qWarning("Unexpected PVCAM acquisition status: %d, bytes=%u", static_cast<int>(status), bytesArrived);
            return false;
        }
        lastStatus = status;
    }
    qWarning("PVCAM readout timeout: lastStatus=%d, bytes=%u", static_cast<int>(lastStatus), bytesArrived);
    return false;
}

QString makeFramePath(const QString& outputDir, const QString& filePrefix, int filePadding, qint64 frameIndex)
{
    const int padding = std::clamp(filePadding, 1, 12);
    return QDir(outputDir).filePath(QStringLiteral("%1%2.tiff")
        .arg(filePrefix)
        .arg(std::max<qint64>(0, frameIndex), padding, 10, QLatin1Char('0')));
}

#endif

} // namespace

PvcamCameraWorker::PvcamCameraWorker(QObject* parent)
    : QObject(parent)
    , queue_(std::make_shared<BoundedFrameQueue>(256))
{
    qRegisterMetaType<ct::CameraFrame>("ct::CameraFrame");
    qRegisterMetaType<ct::PvcamCaptureSettings>("ct::PvcamCaptureSettings");
    qRegisterMetaType<ct::PvcamTimingCalibrationSettings>("ct::PvcamTimingCalibrationSettings");
    qRegisterMetaType<ct::PvcamTimingCalibrationResult>("ct::PvcamTimingCalibrationResult");
    qRegisterMetaType<ct::ContinuousTimingStats>("ct::ContinuousTimingStats");
}

void PvcamCameraWorker::setContinuousFrameSink(ContinuousFrameSink* sink)
{
    continuousFrameSink_ = sink;
}

bool PvcamCameraWorker::ensureFramePool(qsizetype frameBytes, int capacity, QString* errorMessage)
{
    const FrameBufferPoolStats current = framePool_.stats();
    if (current.frameBytes == frameBytes && current.capacity == capacity)
    {
        return true;
    }
    QElapsedTimer timer;
    timer.start();
    if (!framePool_.configure(frameBytes, capacity, true, errorMessage))
    {
        return false;
    }
    const FrameBufferPoolStats configured = framePool_.stats();
    emit logMessage(QStringLiteral("Kinetix frame pool allocated: %1 slots, %2 MiB/frame, %3 GiB total, %4 ms.")
        .arg(configured.capacity)
        .arg(static_cast<double>(configured.frameBytes) / (1024.0 * 1024.0), 0, 'f', 2)
        .arg(static_cast<double>(configured.totalBytes) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 2)
        .arg(timer.elapsed()));
    return true;
}

void PvcamCameraWorker::armContinuousFrameGate()
{
    const qint64 nowNs = static_cast<qint64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    continuousGateArmSteadyNs_.store(nowNs, std::memory_order_release);
    continuousGateArmed_.store(true, std::memory_order_release);
}

#ifdef CT_HAS_PVCAM
void PvcamCameraWorker::cleanupPreparedSequence(bool abortExposure)
{
    if (!preparedActive_)
    {
        return;
    }
    if (abortExposure && hcam_ >= 0)
    {
        pl_exp_abort(hcam_, CCS_HALT);
    }
    pl_exp_uninit_seq();
    preparedFrameBuffer_.clear();
    preparedActive_ = false;
    preparedFrameRunning_ = false;
    preparedFrameIndex_ = 0;
}
#endif

void PvcamCameraWorker::openCameraIfNeeded()
{
#ifndef CT_HAS_PVCAM
    throw std::runtime_error("PVCAM SDK was not found at build time.");
#else
    writePvcamTrace(QStringLiteral("PVCAM openCameraIfNeeded enter"));
    if (cameraOpened_)
    {
        writePvcamTrace(QStringLiteral("PVCAM hot session already open"));
        emit logMessage(QStringLiteral("Kinetix using hot camera session: %1").arg(cameraName_));
        return;
    }

    QElapsedTimer openTimer;
    openTimer.start();
    emit cameraStateChanged(QStringLiteral("Opening"));

    if (!pvcamInitialized_)
    {
        writePvcamTrace(QStringLiteral("PVCAM pl_pvcam_init begin"));
        if (PV_OK != pl_pvcam_init())
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        pvcamInitialized_ = true;
        writePvcamTrace(QStringLiteral("PVCAM pl_pvcam_init ok"));
    }

    int16 total = 0;
    writePvcamTrace(QStringLiteral("PVCAM pl_cam_get_total begin"));
    if (PV_OK != pl_cam_get_total(&total) || total <= 0)
    {
        throw std::runtime_error("No PVCAM cameras found.");
    }
    writePvcamTrace(QStringLiteral("PVCAM pl_cam_get_total ok: %1").arg(total));

    char cameraName[CAM_NAME_LEN] = {};
    writePvcamTrace(QStringLiteral("PVCAM pl_cam_get_name begin"));
    if (PV_OK != pl_cam_get_name(0, cameraName))
    {
        throw std::runtime_error(pvcamError().toStdString());
    }
    writePvcamTrace(QStringLiteral("PVCAM pl_cam_open begin: %1").arg(QString::fromLatin1(cameraName)));
    if (PV_OK != pl_cam_open(cameraName, &hcam_, OPEN_EXCLUSIVE))
    {
        throw std::runtime_error(pvcamError().toStdString());
    }
    cameraOpened_ = true;
    cameraName_ = QString::fromLatin1(cameraName);
    const QString readoutSummary = initializeKinetixReadout(hcam_);

    uns16 sensorWidth = 0;
    uns16 sensorHeight = 0;
    if (!getParam(hcam_, PARAM_SER_SIZE, ATTR_CURRENT, sensorWidth)
        || !getParam(hcam_, PARAM_PAR_SIZE, ATTR_CURRENT, sensorHeight))
    {
        throw std::runtime_error("Failed to read PVCAM sensor size.");
    }
    sensorWidth_ = sensorWidth;
    sensorHeight_ = sensorHeight;

    int32 imageFormat = 0;
    if (!getParam(hcam_, PARAM_IMAGE_FORMAT, ATTR_CURRENT, imageFormat))
    {
        throw std::runtime_error("Failed to read PVCAM image format.");
    }
    imageFormat_ = imageFormat;

    int16 bitDepth = 0;
    getParam(hcam_, PARAM_BIT_DEPTH, ATTR_CURRENT, bitDepth);
    bitDepth_ = bitDepth;

    emit logMessage(QStringLiteral("Kinetix opened hot session: %1 (%2 ms).")
        .arg(cameraName_)
        .arg(openTimer.elapsed()));
    if (!readoutSummary.isEmpty())
    {
        emit logMessage(QStringLiteral("Kinetix initialized readout: %1.").arg(readoutSummary));
    }
    writePvcamTrace(QStringLiteral("PVCAM openCameraIfNeeded ok"));
#endif
}

void PvcamCameraWorker::closeCamera()
{
#ifdef CT_HAS_PVCAM
    writePvcamTrace(QStringLiteral("PVCAM closeCamera enter"));
    writePvcamTrace(QStringLiteral("PVCAM closeCamera cleanupPreparedSequence begin"));
    cleanupPreparedSequence(true);
    writePvcamTrace(QStringLiteral("PVCAM closeCamera cleanupPreparedSequence end"));
    if (cameraOpened_)
    {
        writePvcamTrace(QStringLiteral("PVCAM pl_cam_close begin"));
        pl_cam_close(hcam_);
        writePvcamTrace(QStringLiteral("PVCAM pl_cam_close end"));
        cameraOpened_ = false;
        hcam_ = -1;
    }
    if (pvcamInitialized_)
    {
        writePvcamTrace(QStringLiteral("PVCAM pl_pvcam_uninit begin"));
        pl_pvcam_uninit();
        writePvcamTrace(QStringLiteral("PVCAM pl_pvcam_uninit end"));
        pvcamInitialized_ = false;
    }
    cameraName_.clear();
    sensorWidth_ = 0;
    sensorHeight_ = 0;
    imageFormat_ = 0;
    bitDepth_ = 0;
    writePvcamTrace(QStringLiteral("PVCAM closeCamera exit"));
#endif
}

void PvcamCameraWorker::warmOpen()
{
#ifndef CT_HAS_PVCAM
    emit cameraStateChanged(QStringLiteral("PVCAM unavailable"));
    emit logMessage(QStringLiteral("PVCAM SDK was not found at build time."));
#else
    writePvcamTrace(QStringLiteral("PVCAM warmOpen enter"));
    if (capturing_.load())
    {
        return;
    }
    try
    {
        openCameraIfNeeded();
        writePvcamTrace(QStringLiteral("PVCAM warmOpen open ok; emitting Ready"));
        emit cameraStateChanged(QStringLiteral("Ready"));
        writePvcamTrace(QStringLiteral("PVCAM warmOpen emitted Ready; exit"));
    }
    catch (const std::exception& ex)
    {
        writePvcamTrace(QStringLiteral("PVCAM warmOpen exception: %1").arg(QString::fromLocal8Bit(ex.what())));
        closeCamera();
        emit cameraStateChanged(QStringLiteral("Error"));
        emit logMessage(QStringLiteral("Kinetix warm open failed: %1").arg(QString::fromLocal8Bit(ex.what())));
        writePvcamTrace(QStringLiteral("PVCAM warmOpen emitted Error; exit"));
    }
#endif
}

void PvcamCameraWorker::startCapture(const PvcamCaptureSettings& rawSettings)
{
    if (preparedActive_)
    {
        emit logMessage(QStringLiteral("Kinetix prepared CT capture is active; stop CT before Live/Snap."));
        emit captureFinished(false, QStringLiteral("Prepared CT capture is active."));
        return;
    }
    if (capturing_.exchange(true))
    {
        emit logMessage(QStringLiteral("Kinetix capture is already running."));
        return;
    }

    PvcamCaptureSettings settings = rawSettings;
    settings.frameCount = std::max(1, settings.frameCount);
    settings.exposureValue = std::max(0.001, settings.exposureValue);
    settings.roiX = std::max(0, settings.roiX);
    settings.roiY = std::max(0, settings.roiY);
    settings.roiWidth = std::max(1, settings.roiWidth);
    settings.roiHeight = std::max(1, settings.roiHeight);
    settings.queueCapacity = std::max(1, settings.queueCapacity);
    settings.framePoolCapacity = std::clamp(settings.framePoolCapacity, 1, 4096);
    settings.filePadding = std::clamp(settings.filePadding, 1, 12);
    if (settings.previewIntervalMs > 0)
    {
        settings.previewIntervalMs = std::clamp<qint64>(settings.previewIntervalMs, 33, 1000);
    }
    if (settings.filePrefix.isEmpty())
    {
        settings.filePrefix = QStringLiteral("kinetix_frame_");
    }
    const double exposureMs = std::max(0.001, exposureMilliseconds(settings.exposureValue, settings.exposureUnit));

    stopRequested_.store(false);
    droppedCount_ = 0;
    queue_->setCapacity(settings.queueCapacity);
    emit captureStatsChanged(0, 0, 0, 0);

#ifndef CT_HAS_PVCAM
    capturing_.store(false);
    emit cameraStateChanged(QStringLiteral("PVCAM unavailable"));
    emit captureFinished(false, QStringLiteral("PVCAM SDK was not found at build time."));
    return;
#else
    bool sequenceInitialized = false;

    try
    {
        QElapsedTimer acquisitionTimer;
        acquisitionTimer.start();
        openCameraIfNeeded();

        const rgn_type roi = makeExplicitRoi(static_cast<uns16>(sensorWidth_), static_cast<uns16>(sensorHeight_), settings);
        const int width = roiWidth(roi);
        const int height = roiHeight(roi);

        const ExposureResolution expResolution = configureExposureResolution(hcam_, settings.exposureUnit);
        const uns32 exposureUnits = std::max<uns32>(
            1,
            static_cast<uns32>(std::llround(exposureMs * expResolution.unitsPerMillisecond)));
        const int16 expMode = selectInternalExposureMode(hcam_);
        uns32 exposureBytes = 0;
        rgn_type mutableRoi = roi;
        if (PV_OK != pl_exp_setup_seq(hcam_, 1, 1, &mutableRoi, expMode,
                exposureUnits, &exposureBytes))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        sequenceInitialized = true;

        int32 activeImageFormat = imageFormat_;
        if (!getParam(hcam_, PARAM_IMAGE_FORMAT_HOST, ATTR_CURRENT, activeImageFormat))
        {
            getParam(hcam_, PARAM_IMAGE_FORMAT, ATTR_CURRENT, activeImageFormat);
        }
        imageFormat_ = activeImageFormat;
        int16 activeBitDepth = bitDepth_;
        if (!getParam(hcam_, PARAM_BIT_DEPTH_HOST, ATTR_CURRENT, activeBitDepth))
        {
            getParam(hcam_, PARAM_BIT_DEPTH, ATTR_CURRENT, activeBitDepth);
        }
        bitDepth_ = activeBitDepth;

        const int cvType = cvTypeForImageFormat(imageFormat_);
        const int bytesPerPixel = bytesPerPixelForCvType(cvType);
        const int strideBytes = width * bytesPerPixel;
        const uns32 expectedBytes = static_cast<uns32>(height * strideBytes);
        if (exposureBytes < expectedBytes)
        {
            throw std::runtime_error("PVCAM returned fewer bytes than the selected ROI requires.");
        }

        emit logMessage(QStringLiteral("Kinetix ROI x=%1 y=%2 w=%3 h=%4, bitDepth=%5, cvType=%6, stride=%7, frameBytes=%8.")
            .arg(roi.s1)
            .arg(roi.p1)
            .arg(width)
            .arg(height)
            .arg(bitDepth_)
            .arg(cvType)
            .arg(strideBytes)
            .arg(exposureBytes));
        emit logMessage(QStringLiteral("Kinetix %1").arg(expResolution.logText));
        emit logMessage(QStringLiteral("Kinetix exposure mode: %1 (%2 %3 = %4 ms, %5 PVCAM units).")
            .arg(exposureModeLabel(expMode))
            .arg(settings.exposureValue)
            .arg(exposureUnitLabel(settings.exposureUnit))
            .arg(exposureMs, 0, 'f', 3)
            .arg(exposureUnits));

        if (settings.saveRaw)
        {
            QDir().mkpath(settings.outputDir);
            if (!QFileInfo(settings.outputDir).isDir())
            {
                throw std::runtime_error("Output directory is unavailable.");
            }
        }

        std::vector<unsigned char> frameBuffer(exposureBytes);
        emit cameraStateChanged(settings.previewOnly ? QStringLiteral("Live") : QStringLiteral("Capturing"));
        emit logMessage(QStringLiteral("Kinetix setup ready in %1 ms.").arg(acquisitionTimer.elapsed()));
        qint64 lastPreviewEmitMs = settings.previewIntervalMs > 0 ? -settings.previewIntervalMs : 0;

        for (int frameIndex = 0; (settings.previewOnly || frameIndex < settings.frameCount) && !stopRequested_.load(); ++frameIndex)
        {
            if (PV_OK != pl_exp_start_seq(hcam_, frameBuffer.data()))
            {
                throw std::runtime_error(pvcamError().toStdString());
            }

            uns32 bytesArrived = 0;
            const uns32 timeoutMs = std::max<uns32>(5000, static_cast<uns32>(std::ceil(exposureMs + 5000.0)));
            if (!waitForReadout(hcam_, timeoutMs, stopRequested_, bytesArrived))
            {
                pl_exp_abort(hcam_, CCS_HALT);
                if (stopRequested_.load())
                {
                    break;
                }
                throw std::runtime_error("Timed out waiting for Kinetix frame; check trigger mode/camera state.");
            }

            cv::Mat wrapped(height, width, cvType, frameBuffer.data(), static_cast<size_t>(strideBytes));
            CameraFrame frame;
            frame.frameIndex = frameIndex;
            frame.timestampMs = QDateTime::currentMSecsSinceEpoch();
            frame.width = width;
            frame.height = height;
            frame.bitDepth = bitDepth_;
            frame.bytesPerPixel = bytesPerPixel;
            frame.strideBytes = strideBytes;
            frame.cvType = cvType;
            frame.image = wrapped.clone();

            if (PV_OK != pl_exp_finish_seq(hcam_, frameBuffer.data(), 0))
            {
                emit logMessage(QStringLiteral("pl_exp_finish_seq warning: %1").arg(pvcamError()));
            }
            if (frameIndex == 0)
            {
                emit logMessage(QStringLiteral("Kinetix first frame acquired in %1 ms.").arg(acquisitionTimer.elapsed()));
            }

            if (!settings.previewOnly && !settings.saveRaw && settings.enqueueFrames && !queue_->push(frame))
            {
                ++droppedCount_;
                throw std::runtime_error("Camera frame queue is full; capture stopped to avoid frame loss.");
            }

            if (!settings.previewOnly && settings.saveRaw)
            {
                const QString framePath = settings.singleFramePath.isEmpty()
                    ? makeFramePath(settings.outputDir, settings.filePrefix, settings.filePadding, frame.frameIndex)
                    : settings.singleFramePath;
                QString writeError;
                if (!writeImageFile(framePath, frame.image, &writeError))
                {
                    throw std::runtime_error(QStringLiteral("Failed to save camera frame: %1")
                        .arg(writeError).toLocal8Bit().constData());
                }
                emit logMessage(QStringLiteral("Kinetix snap saved: %1.").arg(framePath));
            }

            const bool isLastFiniteCaptureFrame = !settings.previewOnly && (frameIndex + 1) >= settings.frameCount;
            const bool shouldEmitPreview =
                frameIndex == 0
                || isLastFiniteCaptureFrame
                || settings.previewIntervalMs <= 0
                || frame.timestampMs - lastPreviewEmitMs >= settings.previewIntervalMs;
            if (shouldEmitPreview)
            {
                lastPreviewEmitMs = frame.timestampMs;
                if (settings.previewOnly && frameIndex == 0)
                {
                    emit logMessage(QStringLiteral("Kinetix live emitting first preview frame: %1x%2, bytes=%3.")
                        .arg(frame.width)
                        .arg(frame.height)
                        .arg(frame.image.total() * frame.image.elemSize()));
                }
                emit frameReady(frame);
            }
            emit captureStatsChanged(frame.frameIndex + 1, queue_->size(), queue_->consumedCount(), droppedCount_);
        }

        if (sequenceInitialized)
        {
            pl_exp_uninit_seq();
            sequenceInitialized = false;
        }

        capturing_.store(false);
        emit cameraStateChanged(QStringLiteral("Ready"));
        emit captureFinished(true, stopRequested_.load() ? QStringLiteral("Capture stopped.") : QStringLiteral("Capture completed."));
    }
    catch (const std::exception& ex)
    {
        if (sequenceInitialized)
        {
            pl_exp_abort(hcam_, CCS_HALT);
            pl_exp_uninit_seq();
        }
        closeCamera();

        capturing_.store(false);
        emit cameraStateChanged(QStringLiteral("Error"));
        emit captureFinished(false, QString::fromLocal8Bit(ex.what()));
    }
#endif
}

void PvcamCameraWorker::beginPreparedCapture(const PvcamCaptureSettings& rawSettings)
{
    if (preparedActive_)
    {
        emit preparedCaptureReady(true, QStringLiteral("Kinetix CT prepared capture is already active."));
        return;
    }
    if (capturing_.exchange(true))
    {
        emit preparedCaptureReady(false, QStringLiteral("Kinetix capture is already running."));
        return;
    }

    PvcamCaptureSettings settings = rawSettings;
    settings.frameCount = 1;
    settings.exposureValue = std::max(0.001, settings.exposureValue);
    settings.roiX = std::max(0, settings.roiX);
    settings.roiY = std::max(0, settings.roiY);
    settings.roiWidth = std::max(1, settings.roiWidth);
    settings.roiHeight = std::max(1, settings.roiHeight);
    settings.queueCapacity = std::max(1, settings.queueCapacity);
    settings.saveRaw = false;
    settings.previewOnly = false;
    settings.enqueueFrames = false;
    settings.previewIntervalMs = 0;

    stopRequested_.store(false);
    droppedCount_ = 0;
    emit captureStatsChanged(0, 0, 0, 0);

#ifndef CT_HAS_PVCAM
    capturing_.store(false);
    emit cameraStateChanged(QStringLiteral("PVCAM unavailable"));
    emit preparedCaptureReady(false, QStringLiteral("PVCAM SDK was not found at build time."));
    return;
#else
    try
    {
        QElapsedTimer setupTimer;
        setupTimer.start();
        openCameraIfNeeded();

        const rgn_type roi = makeExplicitRoi(static_cast<uns16>(sensorWidth_), static_cast<uns16>(sensorHeight_), settings);
        preparedWidth_ = roiWidth(roi);
        preparedHeight_ = roiHeight(roi);
        preparedSettings_ = settings;
        preparedExposureMs_ = std::max(0.001, exposureMilliseconds(settings.exposureValue, settings.exposureUnit));

        const ExposureResolution expResolution = configureExposureResolution(hcam_, settings.exposureUnit);
        const uns32 exposureUnits = std::max<uns32>(
            1,
            static_cast<uns32>(std::llround(preparedExposureMs_ * expResolution.unitsPerMillisecond)));
        preparedExposureMode_ = selectInternalExposureMode(hcam_);
        preparedExposureBytes_ = 0;
        rgn_type mutableRoi = roi;
        if (PV_OK != pl_exp_setup_seq(hcam_, 1, 1, &mutableRoi, preparedExposureMode_,
                exposureUnits, &preparedExposureBytes_))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        preparedActive_ = true;

        int32 activeImageFormat = imageFormat_;
        if (!getParam(hcam_, PARAM_IMAGE_FORMAT_HOST, ATTR_CURRENT, activeImageFormat))
        {
            getParam(hcam_, PARAM_IMAGE_FORMAT, ATTR_CURRENT, activeImageFormat);
        }
        imageFormat_ = activeImageFormat;
        int16 activeBitDepth = bitDepth_;
        if (!getParam(hcam_, PARAM_BIT_DEPTH_HOST, ATTR_CURRENT, activeBitDepth))
        {
            getParam(hcam_, PARAM_BIT_DEPTH, ATTR_CURRENT, activeBitDepth);
        }
        bitDepth_ = activeBitDepth;

        preparedCvType_ = cvTypeForImageFormat(imageFormat_);
        preparedBytesPerPixel_ = bytesPerPixelForCvType(preparedCvType_);
        preparedStrideBytes_ = preparedWidth_ * preparedBytesPerPixel_;
        const uns32 expectedBytes = static_cast<uns32>(preparedHeight_ * preparedStrideBytes_);
        if (preparedExposureBytes_ < expectedBytes)
        {
            throw std::runtime_error("PVCAM returned fewer bytes than the selected ROI requires.");
        }
        QString poolError;
        if (!ensureFramePool(static_cast<qsizetype>(expectedBytes), settings.framePoolCapacity, &poolError))
        {
            throw std::runtime_error(("Failed to prepare CT frame pool: " + poolError).toStdString());
        }
        preparedFrameBuffer_.assign(preparedExposureBytes_, 0);
        preparedFrameIndex_ = 0;

        emit cameraStateChanged(QStringLiteral("CT Prepared"));
        emit logMessage(QStringLiteral("Kinetix CT prepared ROI x=%1 y=%2 w=%3 h=%4, bitDepth=%5, cvType=%6, stride=%7, frameBytes=%8.")
            .arg(roi.s1)
            .arg(roi.p1)
            .arg(preparedWidth_)
            .arg(preparedHeight_)
            .arg(bitDepth_)
            .arg(preparedCvType_)
            .arg(preparedStrideBytes_)
            .arg(preparedExposureBytes_));
        emit logMessage(QStringLiteral("Kinetix CT prepared %1").arg(expResolution.logText));
        emit logMessage(QStringLiteral("Kinetix CT prepared exposure mode: %1 (%2 %3 = %4 ms, %5 PVCAM units).")
            .arg(exposureModeLabel(preparedExposureMode_))
            .arg(settings.exposureValue)
            .arg(exposureUnitLabel(settings.exposureUnit))
            .arg(preparedExposureMs_, 0, 'f', 3)
            .arg(exposureUnits));
        emit logMessage(QStringLiteral("Kinetix CT prepared setup ready in %1 ms.").arg(setupTimer.elapsed()));
        emit preparedCaptureReady(true, QStringLiteral("Kinetix CT prepared capture ready."));
    }
    catch (const std::exception& ex)
    {
        cleanupPreparedSequence(true);
        closeCamera();
        capturing_.store(false);
        emit cameraStateChanged(QStringLiteral("Error"));
        emit preparedCaptureReady(false, QString::fromLocal8Bit(ex.what()));
    }
#endif
}

void PvcamCameraWorker::capturePreparedFrame(int frameCount)
{
    frameCount = std::max(1, frameCount);
#ifndef CT_HAS_PVCAM
    emit preparedFrameBatchFinished(false, QStringLiteral("PVCAM SDK was not found at build time."));
    return;
#else
    if (!preparedActive_)
    {
        emit preparedFrameBatchFinished(false, QStringLiteral("Kinetix CT prepared capture is not active."));
        return;
    }
    if (preparedFrameRunning_)
    {
        emit preparedFrameBatchFinished(false, QStringLiteral("Kinetix CT prepared frame capture is already running."));
        return;
    }
    preparedFrameRunning_ = true;
    stopRequested_.store(false);

    try
    {
        QElapsedTimer frameTimer;
        frameTimer.start();
        int capturedThisBatch = 0;
        for (int i = 0; i < frameCount && !stopRequested_.load(); ++i)
        {
            if (PV_OK != pl_exp_start_seq(hcam_, preparedFrameBuffer_.data()))
            {
                throw std::runtime_error(pvcamError().toStdString());
            }

            uns32 bytesArrived = 0;
            const uns32 timeoutMs = std::max<uns32>(5000, static_cast<uns32>(std::ceil(preparedExposureMs_ + 5000.0)));
            if (!waitForReadout(hcam_, timeoutMs, stopRequested_, bytesArrived))
            {
                pl_exp_abort(hcam_, CCS_HALT);
                if (stopRequested_.load())
                {
                    break;
                }
                throw std::runtime_error("Timed out waiting for Kinetix CT prepared frame; check trigger mode/camera state.");
            }

            FrameBufferPool::LeasePtr lease = framePool_.tryAcquire();
            if (!lease)
            {
                throw std::runtime_error("FramePoolExhausted during prepared CT capture.");
            }
            const qsizetype imageBytes = static_cast<qsizetype>(preparedHeight_) * preparedStrideBytes_;
            std::memcpy(lease->data(), preparedFrameBuffer_.data(), static_cast<size_t>(imageBytes));
            CameraFrame frame;
            frame.frameIndex = preparedFrameIndex_;
            frame.timestampMs = QDateTime::currentMSecsSinceEpoch();
            frame.width = preparedWidth_;
            frame.height = preparedHeight_;
            frame.bitDepth = bitDepth_;
            frame.bytesPerPixel = preparedBytesPerPixel_;
            frame.strideBytes = preparedStrideBytes_;
            frame.cvType = preparedCvType_;
            frame.bufferLease = lease;

            if (PV_OK != pl_exp_finish_seq(hcam_, preparedFrameBuffer_.data(), 0))
            {
                emit logMessage(QStringLiteral("pl_exp_finish_seq warning during CT prepared capture: %1").arg(pvcamError()));
            }

            emit frameReady(frame);
            ++preparedFrameIndex_;
            ++capturedThisBatch;
            emit captureStatsChanged(preparedFrameIndex_, queue_->size(), queue_->consumedCount(), droppedCount_);
        }
        preparedFrameRunning_ = false;
        emit preparedFrameBatchFinished(!stopRequested_.load(),
            stopRequested_.load()
                ? QStringLiteral("Kinetix CT prepared capture stopped.")
                : QStringLiteral("Kinetix CT prepared captured %1 frame(s) in %2 ms.").arg(capturedThisBatch).arg(frameTimer.elapsed()));
    }
    catch (const std::exception& ex)
    {
        preparedFrameRunning_ = false;
        cleanupPreparedSequence(true);
        closeCamera();
        capturing_.store(false);
        emit cameraStateChanged(QStringLiteral("Error"));
        emit preparedFrameBatchFinished(false, QString::fromLocal8Bit(ex.what()));
    }
#endif
}

void PvcamCameraWorker::endPreparedCapture()
{
#ifndef CT_HAS_PVCAM
    capturing_.store(false);
    return;
#else
    const bool hadPrepared = preparedActive_;
    cleanupPreparedSequence(stopRequested_.load());
    capturing_.store(false);
    if (hadPrepared)
    {
        emit cameraStateChanged(QStringLiteral("Ready"));
        emit logMessage(QStringLiteral("Kinetix CT prepared capture ended."));
    }
#endif
}

void PvcamCameraWorker::calibrateContinuousTiming(const PvcamTimingCalibrationSettings& rawSettings)
{
    PvcamTimingCalibrationResult result;
    if (preparedActive_)
    {
        result.message = QStringLiteral("Prepared CT capture is active.");
        emit timingCalibrationFinished(result);
        return;
    }
    if (capturing_.exchange(true))
    {
        result.message = QStringLiteral("Kinetix capture is already running.");
        emit timingCalibrationFinished(result);
        return;
    }

    PvcamTimingCalibrationSettings settings = rawSettings;
    settings.exposureValue = std::max(0.001, settings.exposureValue);
    settings.roiX = std::max(0, settings.roiX);
    settings.roiY = std::max(0, settings.roiY);
    settings.roiWidth = std::max(1, settings.roiWidth);
    settings.roiHeight = std::max(1, settings.roiHeight);
    settings.warmupFrames = std::clamp(settings.warmupFrames, 0, 10000);
    settings.sampleFrames = std::clamp(settings.sampleFrames, 3, 10000);
    settings.ringFrameCount = std::clamp(settings.ringFrameCount, 4, 256);
    stopRequested_.store(false);

#ifndef CT_HAS_PVCAM
    capturing_.store(false);
    result.message = QStringLiteral("PVCAM SDK was not found at build time.");
    emit timingCalibrationFinished(result);
#else
    bool continuousStarted = false;
    bool metadataEnabled = false;
    FRAME_INFO* frameInfo = nullptr;
    md_frame* metadataFrame = nullptr;
    auto cleanup = [&]() {
        if (continuousStarted)
        {
            pl_exp_stop_cont(hcam_, CCS_HALT);
            continuousStarted = false;
        }
        if (metadataFrame != nullptr)
        {
            pl_md_release_frame_struct(metadataFrame);
            metadataFrame = nullptr;
        }
        if (frameInfo != nullptr)
        {
            pl_release_frame_info_struct(frameInfo);
            frameInfo = nullptr;
        }
        if (metadataEnabled && cameraOpened_ && hcam_ >= 0)
        {
            rs_bool disabled = FALSE;
            pl_set_param(hcam_, PARAM_METADATA_ENABLED, &disabled);
            metadataEnabled = false;
        }
    };

    try
    {
        QElapsedTimer timer;
        timer.start();
        openCameraIfNeeded();

        PvcamCaptureSettings roiSettings;
        roiSettings.roiX = settings.roiX;
        roiSettings.roiY = settings.roiY;
        roiSettings.roiWidth = settings.roiWidth;
        roiSettings.roiHeight = settings.roiHeight;
        const rgn_type roi = makeExplicitRoi(static_cast<uns16>(sensorWidth_), static_cast<uns16>(sensorHeight_), roiSettings);
        result.roiX = roi.s1;
        result.roiY = roi.p1;
        result.roiWidth = roiWidth(roi);
        result.roiHeight = roiHeight(roi);
        result.exposureMs = std::max(0.001, exposureMilliseconds(settings.exposureValue, settings.exposureUnit));
        result.cameraName = cameraName_;
        result.readoutMode = QStringLiteral("16-bit Dynamic Range");
        result.triggerMode = QStringLiteral("internal_timed");
        result.binningX = 1;
        result.binningY = 1;
        if (paramAvailable(hcam_, PARAM_CAMERA_PART_NUMBER))
        {
            result.cameraPartNumber = readStringParam(hcam_, PARAM_CAMERA_PART_NUMBER, MAX_CAM_PART_NUM_LEN);
        }
        if (paramAvailable(hcam_, PARAM_HEAD_SER_NUM_ALPHA))
        {
            result.cameraSerial = readStringParam(hcam_, PARAM_HEAD_SER_NUM_ALPHA, MAX_ALPHA_SER_NUM_LEN);
        }
        if (paramAvailable(hcam_, PARAM_READOUT_PORT))
        {
            result.readoutPort = currentEnumValueName(hcam_, PARAM_READOUT_PORT);
        }
        if (paramAvailable(hcam_, PARAM_SPDTAB_NAME))
        {
            result.readoutSpeed = readStringParam(hcam_, PARAM_SPDTAB_NAME, MAX_SPDTAB_NAME_LEN);
        }
        if (paramAvailable(hcam_, PARAM_GAIN_NAME))
        {
            result.gain = readStringParam(hcam_, PARAM_GAIN_NAME, MAX_GAIN_NAME_LEN);
        }

        const ExposureResolution expResolution = configureExposureResolution(hcam_, settings.exposureUnit);
        const uns32 exposureUnits = std::max<uns32>(
            1,
            static_cast<uns32>(std::llround(result.exposureMs * expResolution.unitsPerMillisecond)));
        const int16 expMode = selectInternalExposureMode(hcam_);

        QString metadataError;
        rs_bool metadataAvailable = FALSE;
        if (PV_OK == pl_get_param(hcam_, PARAM_METADATA_ENABLED, ATTR_AVAIL, &metadataAvailable)
            && metadataAvailable == TRUE)
        {
            rs_bool enabled = TRUE;
            metadataEnabled = setParamIfAvailable(hcam_, PARAM_METADATA_ENABLED, enabled, &metadataError);
        }

        uns32 frameBytes = 0;
        if (PV_OK != pl_exp_setup_cont(hcam_, 1, &roi, expMode, exposureUnits, &frameBytes, CIRC_NO_OVERWRITE))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }

        int16 activeBitDepth = bitDepth_;
        if (!getParam(hcam_, PARAM_BIT_DEPTH_HOST, ATTR_CURRENT, activeBitDepth))
        {
            getParam(hcam_, PARAM_BIT_DEPTH, ATTR_CURRENT, activeBitDepth);
        }
        bitDepth_ = activeBitDepth;
        result.bitDepth = bitDepth_;

        const quint64 maxRingFramesByApi = frameBytes > 0
            ? (static_cast<quint64>(std::numeric_limits<uns32>::max()) - 1ULL) / static_cast<quint64>(frameBytes)
            : 0ULL;
        const int ringFrameCount = std::max(4, std::min(settings.ringFrameCount,
            static_cast<int>(std::min<quint64>(maxRingFramesByApi, static_cast<quint64>(std::numeric_limits<int>::max())))));
        const quint64 ringBytes64 = static_cast<quint64>(frameBytes) * static_cast<quint64>(ringFrameCount);
        if (ringBytes64 > std::numeric_limits<uns32>::max())
        {
            throw std::runtime_error("Timing calibration ring exceeds the PVCAM 32-bit buffer limit.");
        }
        std::vector<unsigned char> ringBuffer(static_cast<size_t>(ringBytes64));

        if (PV_OK != pl_create_frame_info_struct(&frameInfo) || frameInfo == nullptr)
        {
            throw std::runtime_error(("Failed to allocate PVCAM FRAME_INFO: " + pvcamError()).toStdString());
        }
        if (metadataEnabled)
        {
            if (PV_OK != pl_md_create_frame_struct_cont(&metadataFrame, 1) || metadataFrame == nullptr)
            {
                throw std::runtime_error(("Failed to allocate PVCAM metadata decoder: " + pvcamError()).toStdString());
            }
        }

        emit cameraStateChanged(QStringLiteral("Timing Calibration"));
        emit logMessage(QStringLiteral("Kinetix timing calibration: exposure=%1 ms, ROI=%2x%3, warmup=%4, samples=%5, metadata=%6, ring=%7 frames.")
            .arg(result.exposureMs, 0, 'f', 4)
            .arg(result.roiWidth)
            .arg(result.roiHeight)
            .arg(settings.warmupFrames)
            .arg(settings.sampleFrames)
            .arg(metadataEnabled ? QStringLiteral("enabled") : QStringLiteral("unavailable"))
            .arg(ringFrameCount));
        if (!metadataEnabled && !metadataError.isEmpty())
        {
            emit logMessage(QStringLiteral("Kinetix embedded metadata unavailable; qualifying driver timing instead: %1")
                .arg(metadataError));
        }

        if (PV_OK != pl_exp_start_cont(hcam_, ringBuffer.data(), static_cast<uns32>(ringBytes64)))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        continuousStarted = true;

        QVector<TimingSample> samples;
        samples.reserve(settings.sampleFrames);
        const int totalFrames = settings.warmupFrames + settings.sampleFrames;
        int receivedFrames = 0;
        qint64 previousTimestampNs = 0;
        qint64 driverTimestampOffsetNs = 0;
        qint64 previousMetadataFrameNumber = 0;
        qint64 previousAcceptedFrameNumber = 0;
        qint64 lastRetrievedDriverFrame = 0;
        bool metadataTimingUsable = metadataEnabled;
        QString metadataFallbackReason;
        while (receivedFrames < totalFrames && !stopRequested_.load())
        {
            int16 status = 0;
            uns32 bytesArrived = 0;
            uns32 bufferCount = 0;
            if (PV_OK != pl_exp_check_cont_status_ex(hcam_, &status, &bytesArrived, &bufferCount, frameInfo))
            {
                throw std::runtime_error(pvcamError().toStdString());
            }

            if (status == FRAME_AVAILABLE || status == READOUT_COMPLETE)
            {
                // FRAME_AVAILABLE is level-triggered. Once the oldest slot is
                // unlocked, the status can remain available until a later driver
                // notification refreshes FRAME_INFO. Do not ask PVCAM for an
                // oldest slot unless the driver delivery waterline advanced.
                const qint64 driverWaterline = frameInfo->FrameNr;
                if (lastRetrievedDriverFrame > 0 && driverWaterline <= lastRetrievedDriverFrame)
                {
                    QThread::msleep(1);
                    continue;
                }
                void* framePtr = nullptr;
                if (PV_OK != pl_exp_get_oldest_frame_ex(hcam_, &framePtr, frameInfo) || framePtr == nullptr)
                {
                    QThread::msleep(1);
                    continue;
                }

                const qint64 driverRetrievedFrame = frameInfo->FrameNr;
                if (lastRetrievedDriverFrame > 0 && driverRetrievedFrame <= lastRetrievedDriverFrame)
                {
                    if (PV_OK != pl_exp_unlock_oldest_frame(hcam_))
                    {
                        throw std::runtime_error(pvcamError().toStdString());
                    }
                    emit logMessage(QStringLiteral("Kinetix timing ignored stale PVCAM ring slot: driver FrameNr %1 after %2.")
                        .arg(driverRetrievedFrame)
                        .arg(lastRetrievedDriverFrame));
                    QThread::msleep(1);
                    continue;
                }
                qint64 frameNumber = frameInfo->FrameNr;
                qint64 timestampNs = 0;
                qint64 exposureTimeNs = 0;
                bool fallbackToDriverTiming = false;
                if (metadataTimingUsable)
                {
                    if (PV_OK != pl_md_frame_decode(metadataFrame, framePtr, frameBytes)
                        || metadataFrame->header == nullptr)
                    {
                        fallbackToDriverTiming = true;
                        metadataFallbackReason = QStringLiteral("embedded metadata decode failed: %1").arg(pvcamError());
                    }
                    else
                    {
                        const md_frame_header* header = metadataFrame->header;
                        result.metadataVersion = header->version;
                        if (header->version >= 3)
                        {
                            const auto* headerV3 = reinterpret_cast<const md_frame_header_v3*>(header);
                            frameNumber = headerV3->frameNr;
                            const quint64 timestampPs = headerV3->timestampBOF > 0
                                ? headerV3->timestampBOF
                                : headerV3->timestampEOF;
                            timestampNs = static_cast<qint64>(timestampPs / 1000ULL);
                            exposureTimeNs = static_cast<qint64>(headerV3->exposureTime / 1000ULL);
                        }
                        else
                        {
                            frameNumber = header->frameNr;
                            const quint64 timestampTicks = header->timestampBOF > 0
                                ? header->timestampBOF
                                : header->timestampEOF;
                            timestampNs = static_cast<qint64>(timestampTicks * static_cast<quint64>(header->timestampResNs));
                            exposureTimeNs = static_cast<qint64>(
                                static_cast<quint64>(header->exposureTime)
                                * static_cast<quint64>(header->exposureTimeResNs));
                        }
                        if (frameNumber <= previousMetadataFrameNumber)
                        {
                            fallbackToDriverTiming = true;
                            metadataFallbackReason = QStringLiteral("embedded metadata frame number is not increasing (%1 after %2)")
                                .arg(frameNumber)
                                .arg(previousMetadataFrameNumber);
                        }
                        else if (timestampNs <= 0 || (previousTimestampNs > 0 && timestampNs <= previousTimestampNs))
                        {
                            fallbackToDriverTiming = true;
                            metadataFallbackReason = QStringLiteral("embedded metadata timestamp is not increasing");
                        }
                        else
                        {
                            previousMetadataFrameNumber = frameNumber;
                        }
                    }
                }
                if (fallbackToDriverTiming)
                {
                    metadataTimingUsable = false;
                    samples.clear();
                    receivedFrames = 0;
                    previousTimestampNs = 0;
                    driverTimestampOffsetNs = 0;
                    previousAcceptedFrameNumber = 0;
                    frameNumber = frameInfo->FrameNr;
                    exposureTimeNs = 0;
                    emit logMessage(QStringLiteral("Kinetix embedded metadata timing rejected; restarting calibration with FRAME_INFO: %1")
                        .arg(metadataFallbackReason));
                }
                if (!metadataTimingUsable)
                {
                    const qint64 timestampTicks = frameInfo->TimeStampBOF > 0
                        ? frameInfo->TimeStampBOF
                        : frameInfo->TimeStamp;
                    timestampNs = timestampTicks * 100000LL;
                }

                if (!metadataTimingUsable && previousAcceptedFrameNumber > 0
                    && frameNumber == previousAcceptedFrameNumber)
                {
                    if (PV_OK != pl_exp_unlock_oldest_frame(hcam_))
                    {
                        throw std::runtime_error(pvcamError().toStdString());
                    }
                    continue;
                }

                if (timestampNs <= 0 || (previousTimestampNs > 0 && timestampNs <= previousTimestampNs))
                {
                    if (driverTimestampOffsetNs == 0)
                    {
                        driverTimestampOffsetNs = timer.nsecsElapsed() - timestampNs;
                    }
                    timestampNs += driverTimestampOffsetNs;
                    if (previousTimestampNs > 0 && timestampNs <= previousTimestampNs)
                    {
                        timestampNs = timer.nsecsElapsed();
                    }
                }
                previousTimestampNs = timestampNs;

                if (PV_OK != pl_exp_unlock_oldest_frame(hcam_))
                {
                    throw std::runtime_error(pvcamError().toStdString());
                }
                lastRetrievedDriverFrame = driverRetrievedFrame;
                if (receivedFrames >= settings.warmupFrames)
                {
                    samples.append({ frameNumber, timestampNs, exposureTimeNs });
                }
                previousAcceptedFrameNumber = frameNumber;
                ++receivedFrames;
                continue;
            }
            if (status != EXPOSURE_IN_PROGRESS && status != READOUT_IN_PROGRESS)
            {
                throw std::runtime_error("Unexpected PVCAM timing calibration status: " + std::to_string(status));
            }
            QThread::msleep(1);
        }

        const bool usedHardwareMetadata = metadataTimingUsable;
        cleanup();
        if (stopRequested_.load())
        {
            throw std::runtime_error("Kinetix timing calibration stopped.");
        }

        result.analysis = analyzeTimingSamples(samples);
        result.timingSource = usedHardwareMetadata
            ? QStringLiteral("embedded_hardware_metadata")
            : QStringLiteral("qualified_driver_timing");
        result.ok = result.analysis.valid;
        result.message = result.ok
            ? QStringLiteral("Kinetix timing ready%1: fitted=%2 ms, %3 fps, max deviation=%4 ms, drift=%5%, "
                "fit residual=%6 ms, exposure=%7..%8 ms, dropped=%9.")
                .arg(result.analysis.stable ? QString() : QStringLiteral(" with warning"))
                .arg(result.analysis.fittedPeriodMs, 0, 'f', 6)
                .arg(result.analysis.rawFrameRateFps, 0, 'f', 3)
                .arg(result.analysis.maxPeriodDeviationMs, 0, 'f', 6)
                .arg(result.analysis.meanDriftPercent, 0, 'f', 4)
                .arg(result.analysis.maxFitResidualMs, 0, 'f', 6)
                .arg(result.analysis.exposureTimingAvailable ? result.analysis.minExposureMs : 0.0, 0, 'f', 6)
                .arg(result.analysis.exposureTimingAvailable ? result.analysis.maxExposureMs : 0.0, 0, 'f', 6)
                .arg(result.analysis.droppedFrames)
            : QStringLiteral("Kinetix timing calibration failed: %1").arg(result.analysis.error);
        capturing_.store(false);
        emit cameraStateChanged(QStringLiteral("Ready"));
        emit logMessage(result.message);
        emit timingCalibrationFinished(result);
    }
    catch (const std::exception& ex)
    {
        cleanup();
        capturing_.store(false);
        result.ok = false;
        result.message = QString::fromLocal8Bit(ex.what());
        emit cameraStateChanged(QStringLiteral("Error"));
        emit logMessage(QStringLiteral("Kinetix timing calibration failed: %1").arg(result.message));
        emit timingCalibrationFinished(result);
    }
#endif
}

void PvcamCameraWorker::startContinuousCapture(const PvcamCaptureSettings& rawSettings)
{
    if (preparedActive_)
    {
        emit captureFinished(false, QStringLiteral("Prepared CT capture is active."));
        return;
    }
    if (capturing_.exchange(true))
    {
        emit logMessage(QStringLiteral("Kinetix capture is already running."));
        return;
    }

    PvcamCaptureSettings settings = rawSettings;
    settings.frameCount = std::max(1, settings.frameCount);
    settings.exposureValue = std::max(0.001, settings.exposureValue);
    settings.roiX = std::max(0, settings.roiX);
    settings.roiY = std::max(0, settings.roiY);
    settings.roiWidth = std::max(1, settings.roiWidth);
    settings.roiHeight = std::max(1, settings.roiHeight);
    settings.queueCapacity = std::max(4, settings.queueCapacity);
    settings.rawFrameLimit = std::max(0, settings.rawFrameLimit);
    settings.saveRaw = false;
    settings.previewOnly = false;
    settings.enqueueFrames = false;
    settings.frameIntervalMs = std::max(0.0, settings.frameIntervalMs);
    settings.firstKeepDelayMs = std::max(0.0, settings.firstKeepDelayMs);
    settings.rawFrameStride = std::clamp(settings.rawFrameStride, 1, 1000000);
    settings.baselineRawFramePeriodMs = std::max(0.0, settings.baselineRawFramePeriodMs);
    settings.framePoolCapacity = std::clamp(settings.framePoolCapacity, 1, 4096);
    settings.ringFrameCount = std::clamp(settings.ringFrameCount, 4, 256);

    stopRequested_.store(false);
    droppedCount_ = 0;
    continuousGateArmed_.store(!settings.waitForFrameGate, std::memory_order_release);
    continuousGateArmSteadyNs_.store(0, std::memory_order_release);
    if (!settings.waitForFrameGate)
    {
        armContinuousFrameGate();
    }
    emit captureStatsChanged(0, 0, 0, 0);

#ifndef CT_HAS_PVCAM
    capturing_.store(false);
    emit cameraStateChanged(QStringLiteral("PVCAM unavailable"));
    emit captureFinished(false, QStringLiteral("PVCAM SDK was not found at build time."));
    return;
#else
    bool continuousStarted = false;
    bool continuousStartNotified = false;
    bool metadataEnabled = false;
    FRAME_INFO* frameInfo = nullptr;
    md_frame* metadataFrame = nullptr;
    ContinuousTimingTracker timingTracker;
    timingTracker.reset(settings.baselineRawFramePeriodMs, settings.rawFrameStride, 16);
    ContinuousFrameSink* frameSink = continuousFrameSink_;
    PvcamRingWaterlineTracker ringTracker;
    try
    {
        QElapsedTimer timer;
        timer.start();
        openCameraIfNeeded();

        rs_bool circularAvailable = FALSE;
        if (PV_OK != pl_get_param(hcam_, PARAM_CIRC_BUFFER, ATTR_AVAIL, &circularAvailable) || circularAvailable != TRUE)
        {
            throw std::runtime_error("PVCAM circular buffer acquisition is not available for this camera/interface.");
        }

        const rgn_type roi = makeExplicitRoi(static_cast<uns16>(sensorWidth_), static_cast<uns16>(sensorHeight_), settings);
        const int width = roiWidth(roi);
        const int height = roiHeight(roi);
        const double exposureMs = std::max(0.001, exposureMilliseconds(settings.exposureValue, settings.exposureUnit));
        const ExposureResolution expResolution = configureExposureResolution(hcam_, settings.exposureUnit);
        const uns32 exposureUnits = std::max<uns32>(
            1,
            static_cast<uns32>(std::llround(exposureMs * expResolution.unitsPerMillisecond)));
        const int16 expMode = selectInternalExposureMode(hcam_);

        QString metadataError;
        rs_bool metadataAvailable = FALSE;
        if (PV_OK == pl_get_param(hcam_, PARAM_METADATA_ENABLED, ATTR_AVAIL, &metadataAvailable)
            && metadataAvailable == TRUE)
        {
            rs_bool enabled = TRUE;
            metadataEnabled = setParamIfAvailable(hcam_, PARAM_METADATA_ENABLED, enabled, &metadataError);
        }

        uns32 frameBytes = 0;
        if (PV_OK != pl_exp_setup_cont(hcam_, 1, &roi, expMode, exposureUnits, &frameBytes, CIRC_NO_OVERWRITE))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }

        uns32 readoutTimeUs = 0;
        getParam(hcam_, PARAM_READOUT_TIME, ATTR_CURRENT, readoutTimeUs);

        int32 activeImageFormat = imageFormat_;
        if (!getParam(hcam_, PARAM_IMAGE_FORMAT_HOST, ATTR_CURRENT, activeImageFormat))
        {
            getParam(hcam_, PARAM_IMAGE_FORMAT, ATTR_CURRENT, activeImageFormat);
        }
        imageFormat_ = activeImageFormat;
        int16 activeBitDepth = bitDepth_;
        if (!getParam(hcam_, PARAM_BIT_DEPTH_HOST, ATTR_CURRENT, activeBitDepth))
        {
            getParam(hcam_, PARAM_BIT_DEPTH, ATTR_CURRENT, activeBitDepth);
        }
        bitDepth_ = activeBitDepth;

        const int cvType = cvTypeForImageFormat(imageFormat_);
        const int bytesPerPixel = bytesPerPixelForCvType(cvType);
        const int strideBytes = width * bytesPerPixel;
        const uns32 expectedBytes = static_cast<uns32>(height * strideBytes);
        if (frameBytes < expectedBytes)
        {
            throw std::runtime_error("PVCAM returned fewer bytes than the selected ROI requires.");
        }
        QString poolError;
        if (!ensureFramePool(static_cast<qsizetype>(expectedBytes), settings.framePoolCapacity, &poolError))
        {
            throw std::runtime_error(("Failed to prepare continuous frame pool: " + poolError).toStdString());
        }

        const quint64 maxRingFramesByApi = frameBytes > 0
            ? (static_cast<quint64>(std::numeric_limits<uns32>::max()) - 1ULL) / static_cast<quint64>(frameBytes)
            : 0ULL;
        const int ringFrameCount = std::max(4, std::min(settings.ringFrameCount,
            static_cast<int>(std::min<quint64>(maxRingFramesByApi, static_cast<quint64>(std::numeric_limits<int>::max())))));
        const quint64 ringBytes64 = static_cast<quint64>(frameBytes) * static_cast<quint64>(ringFrameCount);
        if (ringBytes64 > std::numeric_limits<uns32>::max())
        {
            throw std::runtime_error("PVCAM continuous ring exceeds the 32-bit buffer limit.");
        }
        std::vector<unsigned char> ringBuffer(static_cast<size_t>(ringBytes64));
        if (PV_OK != pl_create_frame_info_struct(&frameInfo) || frameInfo == nullptr)
        {
            throw std::runtime_error(("Failed to allocate PVCAM FRAME_INFO: " + pvcamError()).toStdString());
        }
        if (metadataEnabled
            && (PV_OK != pl_md_create_frame_struct_cont(&metadataFrame, 1) || metadataFrame == nullptr))
        {
            throw std::runtime_error(("Failed to allocate PVCAM metadata decoder: " + pvcamError()).toStdString());
        }

        emit cameraStateChanged(QStringLiteral("CT Continuous"));
        ringTracker.reset(ringFrameCount);
        emit continuousRingStatsChanged(0, ringFrameCount, 0);
        emit logMessage(QStringLiteral("Kinetix CT continuous ROI x=%1 y=%2 w=%3 h=%4, bitDepth=%5, frameBytes=%6, ring=%7 frames.")
            .arg(roi.s1)
            .arg(roi.p1)
            .arg(width)
            .arg(height)
            .arg(bitDepth_)
            .arg(frameBytes)
            .arg(ringFrameCount));
        emit logMessage(QStringLiteral("Kinetix CT continuous exposure mode: %1 (%2 %3 = %4 ms, %5 PVCAM units).")
            .arg(exposureModeLabel(expMode))
            .arg(settings.exposureValue)
            .arg(exposureUnitLabel(settings.exposureUnit))
            .arg(exposureMs, 0, 'f', 3)
            .arg(exposureUnits));
        emit logMessage(QStringLiteral("Kinetix continuous interval strategy: camera internal free-run plus fixed device-frame stride %1; readout estimate=%2 ms.")
            .arg(settings.rawFrameStride)
            .arg(static_cast<double>(readoutTimeUs) / 1000.0, 0, 'f', 4));
        emit logMessage(metadataEnabled
            ? QStringLiteral("Kinetix continuous timing source: embedded hardware metadata (FrameNr/BOF).")
            : QStringLiteral("Kinetix continuous hardware metadata unavailable; using FRAME_INFO driver timing: %1")
                .arg(metadataError.isEmpty() ? QStringLiteral("PARAM_METADATA_ENABLED unavailable") : metadataError));
        emit continuousIntervalStrategyResolved(QStringLiteral("device_frame_stride"), QString());

        if (PV_OK != pl_exp_start_cont(hcam_, ringBuffer.data(), static_cast<uns32>(ringBytes64)))
        {
            throw std::runtime_error(pvcamError().toStdString());
        }
        continuousStarted = true;
        emit logMessage(QStringLiteral("Kinetix CT continuous started in %1 ms.").arg(timer.elapsed()));
        emit continuousCaptureStarted(true, QStringLiteral("Kinetix CT continuous started."));
        continuousStartNotified = true;

        qint64 rawCaptured = 0;
        qint64 emitted = 0;
        qint64 discarded = 0;
        qint64 firstKeepDeviceFrame = 0;
        qint64 firstKeepTimestampNs = 0;
        qint64 lastRawFrameReceivedNs = timer.nsecsElapsed();
        qint64 lastTimingStatsEmitMs = -100;
        qint64 lastRetrievedDriverFrame = 0;
        const qint64 noFrameTimeoutNs = static_cast<qint64>(
            std::max(10000.0, exposureMs * 3.0 + 5000.0) * 1000000.0);
        while (emitted < settings.frameCount && !stopRequested_.load())
        {
            const qint64 nowSteadyNs = static_cast<qint64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            if (settings.waitForFrameGate && !continuousGateArmed_.load(std::memory_order_acquire)
                && timer.elapsed() > 60000)
            {
                throw std::runtime_error("Timed out waiting for the Fly motor frame gate.");
            }
            if (timer.nsecsElapsed() - lastRawFrameReceivedNs > noFrameTimeoutNs)
            {
                throw std::runtime_error("Kinetix continuous acquisition stopped delivering raw frames.");
            }

            int16 status = 0;
            uns32 bytesArrived = 0;
            uns32 bufferCount = 0;
            if (PV_OK != pl_exp_check_cont_status_ex(hcam_, &status, &bytesArrived, &bufferCount, frameInfo))
            {
                throw std::runtime_error(pvcamError().toStdString());
            }

            if (status == FRAME_AVAILABLE || status == READOUT_COMPLETE)
            {
                // FRAME_AVAILABLE stays asserted while the continuous ring has
                // data. FRAME_INFO from check_cont_status_ex is the driver-side
                // delivery waterline, independent of the camera metadata used
                // for hardware BOF timing below. Waiting for it to advance keeps
                // get_oldest from revisiting a stale slot after a ring wrap.
                const qint64 driverWaterline = frameInfo->FrameNr;
                const PvcamRingEstimate ringBeforeRead = ringTracker.update(driverWaterline, lastRetrievedDriverFrame);
                emit continuousRingStatsChanged(ringBeforeRead.unread, ringBeforeRead.capacity, ringBeforeRead.peakUnread);
                if (lastRetrievedDriverFrame > 0 && driverWaterline <= lastRetrievedDriverFrame)
                {
                    QThread::msleep(1);
                    continue;
                }
                void* framePtr = nullptr;
                if (PV_OK != pl_exp_get_oldest_frame_ex(hcam_, &framePtr, frameInfo) || framePtr == nullptr)
                {
                    QThread::msleep(1);
                    continue;
                }
                const qint64 driverRetrievedFrame = frameInfo->FrameNr;
                if (lastRetrievedDriverFrame > 0 && driverRetrievedFrame <= lastRetrievedDriverFrame)
                {
                    if (PV_OK != pl_exp_unlock_oldest_frame(hcam_))
                    {
                        throw std::runtime_error(pvcamError().toStdString());
                    }
                    emit logMessage(QStringLiteral("Kinetix ignored stale PVCAM ring slot: driver FrameNr %1 after %2.")
                        .arg(driverRetrievedFrame)
                        .arg(lastRetrievedDriverFrame));
                    QThread::msleep(1);
                    continue;
                }
                lastRawFrameReceivedNs = timer.nsecsElapsed();
                qint64 deviceFrame = frameInfo->FrameNr;
                const qint64 bofTicks = frameInfo->TimeStampBOF > 0 ? frameInfo->TimeStampBOF : frameInfo->TimeStamp;
                qint64 bofTimestampNs = bofTicks > 0 ? bofTicks * 100000LL : 0;
                qint64 eofTimestampNs = frameInfo->TimeStamp > 0 ? frameInfo->TimeStamp * 100000LL : bofTimestampNs;
                const void* imageData = framePtr;
                QString frameTimingSource = QStringLiteral("qualified_driver_timing_device_frame_stride");
                if (metadataEnabled)
                {
                    if (PV_OK != pl_md_frame_decode(metadataFrame, framePtr, frameBytes)
                        || metadataFrame->header == nullptr
                        || metadataFrame->roiCount != 1
                        || metadataFrame->roiArray == nullptr
                        || metadataFrame->roiArray[0].data == nullptr
                        || metadataFrame->roiArray[0].dataSize < expectedBytes)
                    {
                        pl_exp_unlock_oldest_frame(hcam_);
                        throw std::runtime_error(("EmbeddedMetadataDecodeFailed: " + pvcamError()).toStdString());
                    }
                    const md_frame_header* header = metadataFrame->header;
                    if (header->version >= 3)
                    {
                        const auto* headerV3 = reinterpret_cast<const md_frame_header_v3*>(header);
                        deviceFrame = headerV3->frameNr;
                        bofTimestampNs = static_cast<qint64>(headerV3->timestampBOF / 1000ULL);
                        eofTimestampNs = static_cast<qint64>(headerV3->timestampEOF / 1000ULL);
                    }
                    else
                    {
                        deviceFrame = header->frameNr;
                        bofTimestampNs = static_cast<qint64>(
                            static_cast<quint64>(header->timestampBOF) * static_cast<quint64>(header->timestampResNs));
                        eofTimestampNs = static_cast<qint64>(
                            static_cast<quint64>(header->timestampEOF) * static_cast<quint64>(header->timestampResNs));
                    }
                    imageData = metadataFrame->roiArray[0].data;
                    frameTimingSource = QStringLiteral("embedded_hardware_metadata_device_frame_stride");
                }
                const ContinuousTimingObservation observation = timingTracker.observeRaw(deviceFrame, bofTimestampNs);
                if (observation == ContinuousTimingObservation::Duplicate)
                {
                    const ContinuousTimingStats duplicateStats = timingTracker.snapshot();
                    if (PV_OK != pl_exp_unlock_oldest_frame(hcam_))
                    {
                        throw std::runtime_error(pvcamError().toStdString());
                    }
                    lastRetrievedDriverFrame = driverRetrievedFrame;
                    if (duplicateStats.duplicateNotifications == 1
                        || duplicateStats.duplicateNotifications % 64 == 0)
                    {
                        emit logMessage(QStringLiteral("Kinetix repeated FrameNr %1 notification ignored (%2 total).")
                            .arg(deviceFrame)
                            .arg(duplicateStats.duplicateNotifications));
                    }
                    continue;
                }
                if (observation == ContinuousTimingObservation::DeviceFrameLoss)
                {
                    const ContinuousTimingStats lossStats = timingTracker.snapshot(true);
                    const FrameBufferPoolStats poolStats = framePool_.stats();
                    emit continuousTimingStatsChanged(lossStats);
                    emit continuousStatsChanged(rawCaptured, emitted, discarded, lossStats.deviceDroppedFrames,
                        poolStats.inUse, poolStats.peakInUse);
                    emit captureStatsChanged(emitted, poolStats.inUse, rawCaptured, lossStats.deviceDroppedFrames);
                    pl_exp_unlock_oldest_frame(hcam_);
                    throw std::runtime_error(QStringLiteral("DeviceFrameLoss: previous=%1 current=%2 missing=%3.")
                        .arg(lossStats.previousDeviceFrame)
                        .arg(lossStats.currentDeviceFrame)
                        .arg(lossStats.missingDeviceFrames)
                        .toStdString());
                }
                if (observation == ContinuousTimingObservation::InvalidTimestamp)
                {
                    const ContinuousTimingStats invalidStats = timingTracker.snapshot(true);
                    emit continuousTimingStatsChanged(invalidStats);
                    pl_exp_unlock_oldest_frame(hcam_);
                    throw std::runtime_error(QStringLiteral("InvalidFrameTimestamp: device frame %1 has a non-increasing BOF timestamp.")
                        .arg(deviceFrame)
                        .toStdString());
                }
                bool keepFrame = false;
                const bool gateArmed = continuousGateArmed_.load(std::memory_order_acquire);
                if (gateArmed)
                {
                    const qint64 gateArmNs = continuousGateArmSteadyNs_.load(std::memory_order_acquire);
                    const double elapsedFromGateMs = gateArmNs > 0
                        ? static_cast<double>(nowSteadyNs - gateArmNs) / 1000000.0
                        : 0.0;
                    if (firstKeepDeviceFrame <= 0 && elapsedFromGateMs >= settings.firstKeepDelayMs)
                    {
                        firstKeepDeviceFrame = deviceFrame;
                        firstKeepTimestampNs = bofTimestampNs;
                        keepFrame = true;
                    }
                    else if (firstKeepDeviceFrame > 0 && deviceFrame > firstKeepDeviceFrame
                        && (deviceFrame - firstKeepDeviceFrame) % settings.rawFrameStride == 0)
                    {
                        keepFrame = true;
                    }
                }

                CameraFrame frame;
                if (keepFrame)
                {
                    FrameBufferPool::LeasePtr lease = framePool_.tryAcquire();
                    if (!lease)
                    {
                        pl_exp_unlock_oldest_frame(hcam_);
                        throw std::runtime_error("FramePoolExhausted during continuous acquisition.");
                    }
                    std::memcpy(lease->data(), imageData, static_cast<size_t>(expectedBytes));
                    frame.frameIndex = rawCaptured;
                    frame.logicalIndex = emitted;
                    frame.deviceFrameNumber = deviceFrame;
                    frame.bofTimestampNs = bofTimestampNs;
                    frame.eofTimestampNs = eofTimestampNs;
                    frame.elapsedMs = firstKeepTimestampNs > 0 && bofTimestampNs >= firstKeepTimestampNs
                        ? static_cast<double>(bofTimestampNs - firstKeepTimestampNs) / 1000000.0
                        : static_cast<double>(timer.elapsed());
                    frame.timestampMs = QDateTime::currentMSecsSinceEpoch();
                    frame.width = width;
                    frame.height = height;
                    frame.bitDepth = bitDepth_;
                    frame.bytesPerPixel = bytesPerPixel;
                    frame.strideBytes = strideBytes;
                    frame.cvType = cvType;
                    frame.timingSource = frameTimingSource;
                    frame.bufferLease = lease;
                }

                if (PV_OK != pl_exp_unlock_oldest_frame(hcam_))
                {
                    throw std::runtime_error(pvcamError().toStdString());
                }
                lastRetrievedDriverFrame = driverRetrievedFrame;
                const PvcamRingEstimate ringAfterRead = ringTracker.update(driverWaterline, lastRetrievedDriverFrame);
                emit continuousRingStatsChanged(ringAfterRead.unread, ringAfterRead.capacity, ringAfterRead.peakUnread);

                if (keepFrame)
                {
                    const ContinuousTimingObservation keptObservation = timingTracker.markKept(deviceFrame, bofTimestampNs);
                    if (keptObservation == ContinuousTimingObservation::KeptFrameStrideViolation)
                    {
                        const ContinuousTimingStats strideStats = timingTracker.snapshot(true);
                        emit continuousTimingStatsChanged(strideStats);
                        throw std::runtime_error(QStringLiteral("KeptFrameStrideViolation: previous=%1 current=%2 expected stride=%3.")
                            .arg(strideStats.previousDeviceFrame)
                            .arg(strideStats.currentDeviceFrame)
                            .arg(settings.rawFrameStride)
                            .toStdString());
                    }
                }

                ++rawCaptured;
                if (keepFrame)
                {
                    if (frameSink != nullptr)
                    {
                        QString sinkError;
                        if (!frameSink->submitContinuousFrame(frame, &sinkError))
                        {
                            throw std::runtime_error(QStringLiteral("Continuous frame sink rejected frame %1: %2")
                                .arg(deviceFrame)
                                .arg(sinkError)
                                .toLocal8Bit().constData());
                        }
                    }
                    else
                    {
                        emit frameReady(frame);
                    }
                    ++emitted;
                }
                else
                {
                    ++discarded;
                }
                const FrameBufferPoolStats poolStats = framePool_.stats();
                const ContinuousTimingStats liveTimingStats = timingTracker.snapshot();
                if (timer.elapsed() - lastTimingStatsEmitMs >= 100)
                {
                    lastTimingStatsEmitMs = timer.elapsed();
                    emit continuousTimingStatsChanged(liveTimingStats);
                }
                if (keepFrame || rawCaptured % 32 == 0)
                {
                    emit continuousStatsChanged(rawCaptured, emitted, discarded, liveTimingStats.deviceDroppedFrames,
                        poolStats.inUse, poolStats.peakInUse);
                }
                emit captureStatsChanged(emitted, poolStats.inUse, rawCaptured, liveTimingStats.deviceDroppedFrames);
                continue;
            }

            if (status != EXPOSURE_IN_PROGRESS && status != READOUT_IN_PROGRESS)
            {
                throw std::runtime_error("Unexpected PVCAM continuous acquisition status: " + std::to_string(status));
            }
            QThread::msleep(1);
        }

        if (continuousStarted)
        {
            pl_exp_stop_cont(hcam_, CCS_HALT);
            continuousStarted = false;
        }
        if (metadataFrame != nullptr)
        {
            pl_md_release_frame_struct(metadataFrame);
            metadataFrame = nullptr;
        }
        if (metadataEnabled)
        {
            rs_bool disabled = FALSE;
            pl_set_param(hcam_, PARAM_METADATA_ENABLED, &disabled);
            metadataEnabled = false;
        }
        if (frameInfo != nullptr)
        {
            pl_release_frame_info_struct(frameInfo);
            frameInfo = nullptr;
        }
        if (!stopRequested_.load() && emitted < settings.frameCount)
        {
            throw std::runtime_error(QStringLiteral("Kinetix continuous ended before enough kept frames: raw=%1 kept=%2 target=%3.")
                .arg(rawCaptured)
                .arg(emitted)
                .arg(settings.frameCount)
                .toStdString());
        }
        capturing_.store(false);
        const FrameBufferPoolStats finalPoolStats = framePool_.stats();
        const ContinuousTimingStats finalTimingStats = timingTracker.snapshot(true);
        emit logMessage(QStringLiteral("Kinetix continuous timing final: raw=%1, kept=%2, baseline=%3 ms, fitted=%4 ms, "
            "rolling drift=%5%, max interval deviation=%6 ms, phase=%7 ms, max phase=%8 ms, duplicates=%9, dropped=%10.")
            .arg(finalTimingStats.rawFrames)
            .arg(finalTimingStats.keptFrames)
            .arg(finalTimingStats.baselinePeriodMs, 0, 'f', 6)
            .arg(finalTimingStats.fittedPeriodMs, 0, 'f', 6)
            .arg(finalTimingStats.rollingDriftPercent, 0, 'f', 4)
            .arg(finalTimingStats.maxIntervalDeviationMs, 0, 'f', 6)
            .arg(finalTimingStats.currentPhaseErrorMs, 0, 'f', 6)
            .arg(finalTimingStats.maxAbsPhaseErrorMs, 0, 'f', 6)
            .arg(finalTimingStats.duplicateNotifications)
            .arg(finalTimingStats.deviceDroppedFrames));
        emit continuousTimingStatsChanged(finalTimingStats);
        emit continuousStatsChanged(rawCaptured, emitted, discarded, finalTimingStats.deviceDroppedFrames,
            finalPoolStats.inUse, finalPoolStats.peakInUse);
        emit captureStatsChanged(emitted, finalPoolStats.inUse, rawCaptured, finalTimingStats.deviceDroppedFrames);
        emit cameraStateChanged(QStringLiteral("Ready"));
        const bool captureOk = !stopRequested_.load();
        const QString captureMessage = captureOk
            ? QStringLiteral("Kinetix continuous captured raw=%1 kept=%2 discarded=%3 frame(s) in %4 ms, stride=%5, duplicate notifications ignored=%6.")
                .arg(rawCaptured).arg(emitted).arg(discarded).arg(timer.elapsed()).arg(settings.rawFrameStride).arg(finalTimingStats.duplicateNotifications)
            : QStringLiteral("Kinetix continuous stopped after raw=%1 kept=%2 frame(s).")
                .arg(rawCaptured).arg(emitted);
        if (frameSink != nullptr)
        {
            frameSink->continuousCaptureFinished(captureOk, captureMessage);
        }
        emit captureFinished(captureOk, captureMessage);
    }
    catch (const std::exception& ex)
    {
        const ContinuousTimingStats failureTimingStats = timingTracker.snapshot(true);
        emit logMessage(QStringLiteral("Kinetix continuous timing at failure: raw=%1, kept=%2, baseline=%3 ms, fitted=%4 ms, "
            "rolling drift=%5%, max interval deviation=%6 ms, phase=%7 ms, max phase=%8 ms, duplicates=%9, dropped=%10.")
            .arg(failureTimingStats.rawFrames)
            .arg(failureTimingStats.keptFrames)
            .arg(failureTimingStats.baselinePeriodMs, 0, 'f', 6)
            .arg(failureTimingStats.fittedPeriodMs, 0, 'f', 6)
            .arg(failureTimingStats.rollingDriftPercent, 0, 'f', 4)
            .arg(failureTimingStats.maxIntervalDeviationMs, 0, 'f', 6)
            .arg(failureTimingStats.currentPhaseErrorMs, 0, 'f', 6)
            .arg(failureTimingStats.maxAbsPhaseErrorMs, 0, 'f', 6)
            .arg(failureTimingStats.duplicateNotifications)
            .arg(failureTimingStats.deviceDroppedFrames));
        emit continuousTimingStatsChanged(failureTimingStats);
        if (continuousStarted)
        {
            pl_exp_stop_cont(hcam_, CCS_HALT);
        }
        if (metadataFrame != nullptr)
        {
            pl_md_release_frame_struct(metadataFrame);
            metadataFrame = nullptr;
        }
        if (metadataEnabled && cameraOpened_ && hcam_ >= 0)
        {
            rs_bool disabled = FALSE;
            pl_set_param(hcam_, PARAM_METADATA_ENABLED, &disabled);
            metadataEnabled = false;
        }
        if (frameInfo != nullptr)
        {
            pl_release_frame_info_struct(frameInfo);
            frameInfo = nullptr;
        }
        closeCamera();
        capturing_.store(false);
        emit cameraStateChanged(QStringLiteral("Error"));
        if (frameSink != nullptr)
        {
            frameSink->continuousCaptureFinished(false, QString::fromLocal8Bit(ex.what()));
        }
        if (!continuousStartNotified)
        {
            emit continuousCaptureStarted(false, QString::fromLocal8Bit(ex.what()));
        }
        emit captureFinished(false, QString::fromLocal8Bit(ex.what()));
    }
#endif
}

void PvcamCameraWorker::stopCapture()
{
    writePvcamTrace(QStringLiteral("PVCAM stopCapture enter"));
    stopRequested_.store(true);
#ifdef CT_HAS_PVCAM
    if (capturing_.load() && cameraOpened_ && hcam_ >= 0)
    {
        writePvcamTrace(QStringLiteral("PVCAM pl_exp_abort begin"));
        pl_exp_abort(hcam_, CCS_HALT);
        writePvcamTrace(QStringLiteral("PVCAM pl_exp_abort end"));
    }
#endif
    writePvcamTrace(QStringLiteral("PVCAM stopCapture exit"));
}

void PvcamCameraWorker::shutdown()
{
    writePvcamTrace(QStringLiteral("PVCAM shutdown enter"));
    stopCapture();
    writePvcamTrace(QStringLiteral("PVCAM shutdown endPreparedCapture begin"));
    endPreparedCapture();
    writePvcamTrace(QStringLiteral("PVCAM shutdown endPreparedCapture end"));
    closeCamera();
    writePvcamTrace(QStringLiteral("PVCAM shutdown exit"));
}

} // namespace ct
