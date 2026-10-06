#include "core/acq/TimingProfileCache.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QSaveFile>

#include <algorithm>

namespace ct
{

namespace
{

QJsonObject timingAnalysisToJson(const TimingAnalysis& analysis)
{
    QJsonObject object;
    object["VALID"] = analysis.valid;
    object["STABLE"] = analysis.stable;
    object["SAMPLE_COUNT"] = analysis.sampleCount;
    object["FIRST_FRAME"] = static_cast<double>(analysis.firstFrameNumber);
    object["LAST_FRAME"] = static_cast<double>(analysis.lastFrameNumber);
    object["DROPPED_FRAMES"] = analysis.droppedFrames;
    object["MEAN_PERIOD_MS"] = analysis.meanPeriodMs;
    object["MEDIAN_PERIOD_MS"] = analysis.medianPeriodMs;
    object["FITTED_PERIOD_MS"] = analysis.fittedPeriodMs;
    object["MIN_PERIOD_MS"] = analysis.minPeriodMs;
    object["MAX_PERIOD_MS"] = analysis.maxPeriodMs;
    object["STDDEV_MS"] = analysis.standardDeviationMs;
    object["MAX_PERIOD_DEVIATION_MS"] = analysis.maxPeriodDeviationMs;
    object["FIRST_HALF_MEAN_PERIOD_MS"] = analysis.firstHalfMeanPeriodMs;
    object["SECOND_HALF_MEAN_PERIOD_MS"] = analysis.secondHalfMeanPeriodMs;
    object["MEAN_DRIFT_MS"] = analysis.meanDriftMs;
    object["MEAN_DRIFT_PERCENT"] = analysis.meanDriftPercent;
    object["MAX_FIT_RESIDUAL_MS"] = analysis.maxFitResidualMs;
    object["EXPOSURE_TIMING_AVAILABLE"] = analysis.exposureTimingAvailable;
    object["MEAN_EXPOSURE_MS"] = analysis.meanExposureMs;
    object["MIN_EXPOSURE_MS"] = analysis.minExposureMs;
    object["MAX_EXPOSURE_MS"] = analysis.maxExposureMs;
    object["MAX_EXPOSURE_DEVIATION_MS"] = analysis.maxExposureDeviationMs;
    object["RAW_FRAME_RATE_FPS"] = analysis.rawFrameRateFps;
    return object;
}

TimingAnalysis timingAnalysisFromJson(const QJsonObject& object)
{
    TimingAnalysis analysis;
    analysis.valid = object.value("VALID").toBool(false);
    analysis.stable = object.value("STABLE").toBool(false);
    analysis.sampleCount = object.value("SAMPLE_COUNT").toInt();
    analysis.firstFrameNumber = static_cast<qint64>(object.value("FIRST_FRAME").toDouble());
    analysis.lastFrameNumber = static_cast<qint64>(object.value("LAST_FRAME").toDouble());
    analysis.droppedFrames = object.value("DROPPED_FRAMES").toInt();
    analysis.meanPeriodMs = object.value("MEAN_PERIOD_MS").toDouble();
    analysis.medianPeriodMs = object.value("MEDIAN_PERIOD_MS").toDouble();
    analysis.fittedPeriodMs = object.value("FITTED_PERIOD_MS").toDouble();
    analysis.minPeriodMs = object.value("MIN_PERIOD_MS").toDouble();
    analysis.maxPeriodMs = object.value("MAX_PERIOD_MS").toDouble();
    analysis.standardDeviationMs = object.value("STDDEV_MS").toDouble();
    analysis.maxPeriodDeviationMs = object.value("MAX_PERIOD_DEVIATION_MS").toDouble();
    analysis.firstHalfMeanPeriodMs = object.value("FIRST_HALF_MEAN_PERIOD_MS").toDouble();
    analysis.secondHalfMeanPeriodMs = object.value("SECOND_HALF_MEAN_PERIOD_MS").toDouble();
    analysis.meanDriftMs = object.value("MEAN_DRIFT_MS").toDouble();
    analysis.meanDriftPercent = object.value("MEAN_DRIFT_PERCENT").toDouble();
    analysis.maxFitResidualMs = object.value("MAX_FIT_RESIDUAL_MS").toDouble();
    analysis.exposureTimingAvailable = object.value("EXPOSURE_TIMING_AVAILABLE").toBool(false);
    analysis.meanExposureMs = object.value("MEAN_EXPOSURE_MS").toDouble();
    analysis.minExposureMs = object.value("MIN_EXPOSURE_MS").toDouble();
    analysis.maxExposureMs = object.value("MAX_EXPOSURE_MS").toDouble();
    analysis.maxExposureDeviationMs = object.value("MAX_EXPOSURE_DEVIATION_MS").toDouble();
    analysis.rawFrameRateFps = object.value("RAW_FRAME_RATE_FPS").toDouble();
    return analysis;
}

QJsonObject profileToJson(const TimingProfile& profile)
{
    QJsonObject key;
    key["CAMERA_IDENTITY"] = profile.key.cameraIdentity;
    key["READOUT_MODE"] = profile.key.readoutMode;
    key["TRIGGER_MODE"] = profile.key.triggerMode;
    key["BINNING_X"] = profile.key.binningX;
    key["BINNING_Y"] = profile.key.binningY;
    key["EXPOSURE_MS"] = profile.key.exposureMs;
    key["ROI_X"] = profile.key.roiX;
    key["ROI_Y"] = profile.key.roiY;
    key["ROI_WIDTH"] = profile.key.roiWidth;
    key["ROI_HEIGHT"] = profile.key.roiHeight;

    QJsonObject object;
    object["KEY"] = profile.key.canonical();
    object["SIGNATURE"] = key;
    object["CAMERA_NAME"] = profile.cameraName;
    object["CAMERA_PART_NUMBER"] = profile.cameraPartNumber;
    object["CAMERA_SERIAL"] = profile.cameraSerial;
    object["BIT_DEPTH"] = profile.bitDepth;
    object["METADATA_VERSION"] = profile.metadataVersion;
    object["TIMING_SOURCE"] = profile.timingSource;
    object["READOUT_PORT"] = profile.readoutPort;
    object["READOUT_SPEED"] = profile.readoutSpeed;
    object["GAIN"] = profile.gain;
    object["ANALYSIS"] = timingAnalysisToJson(profile.analysis);
    object["CALIBRATED_AT"] = profile.calibratedAt;
    object["LAST_USED_AT"] = profile.lastUsedAt;
    return object;
}

bool profileFromJson(const QJsonObject& object, TimingProfile* profile)
{
    if (profile == nullptr)
    {
        return false;
    }
    const QJsonObject key = object.value("SIGNATURE").toObject();
    TimingProfile result;
    result.key.cameraIdentity = key.value("CAMERA_IDENTITY").toString();
    result.key.readoutMode = key.value("READOUT_MODE").toString();
    result.key.triggerMode = key.value("TRIGGER_MODE").toString();
    result.key.binningX = key.value("BINNING_X").toInt(1);
    result.key.binningY = key.value("BINNING_Y").toInt(1);
    result.key.exposureMs = key.value("EXPOSURE_MS").toDouble();
    result.key.roiX = key.value("ROI_X").toInt();
    result.key.roiY = key.value("ROI_Y").toInt();
    result.key.roiWidth = key.value("ROI_WIDTH").toInt();
    result.key.roiHeight = key.value("ROI_HEIGHT").toInt();
    result.cameraName = object.value("CAMERA_NAME").toString();
    result.cameraPartNumber = object.value("CAMERA_PART_NUMBER").toString();
    result.cameraSerial = object.value("CAMERA_SERIAL").toString();
    result.bitDepth = object.value("BIT_DEPTH").toInt();
    result.metadataVersion = object.value("METADATA_VERSION").toInt();
    result.timingSource = object.value("TIMING_SOURCE").toString();
    result.readoutPort = object.value("READOUT_PORT").toString();
    result.readoutSpeed = object.value("READOUT_SPEED").toString();
    result.gain = object.value("GAIN").toString();
    result.analysis = timingAnalysisFromJson(object.value("ANALYSIS").toObject());
    result.calibratedAt = object.value("CALIBRATED_AT").toString();
    result.lastUsedAt = object.value("LAST_USED_AT").toString();
    if (result.key.canonical().isEmpty() || !result.analysis.valid || result.analysis.fittedPeriodMs <= 0.0)
    {
        return false;
    }
    *profile = std::move(result);
    return true;
}

} // namespace

QString TimingProfileKey::canonical() const
{
    return QStringLiteral("camera=%1|readout=%2|trigger=%3|binning=%4x%5|exposureMs=%6|roi=%7,%8,%9,%10")
        .arg(cameraIdentity.trimmed(), readoutMode.trimmed(), triggerMode.trimmed())
        .arg(binningX)
        .arg(binningY)
        .arg(exposureMs, 0, 'f', 9)
        .arg(roiX)
        .arg(roiY)
        .arg(roiWidth)
        .arg(roiHeight);
}

bool TimingProfileCache::load(const QString& filePath, QString* errorMessage)
{
    clear();
    QFile file(filePath);
    if (!file.exists())
    {
        return true;
    }
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage) { *errorMessage = file.errorString(); }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        if (errorMessage) { *errorMessage = parseError.errorString(); }
        return false;
    }
    if (!document.object().value("PROFILES").isArray())
    {
        if (errorMessage) { *errorMessage = QStringLiteral("Timing cache root has no PROFILES array."); }
        return false;
    }
    const QJsonArray entries = document.object().value("PROFILES").toArray();
    int entryIndex = 0;
    for (const QJsonValue& value : entries)
    {
        TimingProfile profile;
        if (!value.isObject() || !profileFromJson(value.toObject(), &profile))
        {
            clear();
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("Invalid timing cache profile at index %1.").arg(entryIndex);
            }
            return false;
        }
        upsert(std::move(profile));
        ++entryIndex;
    }
    return true;
}

bool TimingProfileCache::saveAtomic(const QString& filePath, QString* errorMessage) const
{
    QJsonArray entries;
    for (const TimingProfile& profile : profiles_)
    {
        entries.append(profileToJson(profile));
    }
    QJsonObject root;
    root["FORMAT_VERSION"] = 1;
    root["PROFILES"] = entries;
    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly))
    {
        if (errorMessage) { *errorMessage = file.errorString(); }
        return false;
    }
    if (file.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) < 0 || !file.commit())
    {
        if (errorMessage) { *errorMessage = file.errorString(); }
        return false;
    }
    return true;
}

const TimingProfile* TimingProfileCache::find(const TimingProfileKey& key) const
{
    const QString canonical = key.canonical();
    for (const TimingProfile& profile : profiles_)
    {
        if (profile.key.canonical() == canonical)
        {
            return &profile;
        }
    }
    return nullptr;
}

bool TimingProfileCache::markUsed(const TimingProfileKey& key, const QString& timestamp)
{
    const QString canonical = key.canonical();
    for (TimingProfile& profile : profiles_)
    {
        if (profile.key.canonical() == canonical)
        {
            profile.lastUsedAt = timestamp;
            return true;
        }
    }
    return false;
}

void TimingProfileCache::upsert(TimingProfile profile)
{
    const QString canonical = profile.key.canonical();
    for (TimingProfile& existing : profiles_)
    {
        if (existing.key.canonical() == canonical)
        {
            existing = std::move(profile);
            return;
        }
    }
    profiles_.append(std::move(profile));
}

void TimingProfileCache::clear()
{
    profiles_.clear();
}

int TimingProfileCache::size() const
{
    return profiles_.size();
}

} // namespace ct
