#include "core/config/AppConfig.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>

#include <algorithm>
#include <type_traits>

namespace ct
{

namespace
{

template <typename T>
T readValue(const QJsonObject& object, const char* key, const T& fallback)
{
    if (!object.contains(key))
    {
        return fallback;
    }

    if constexpr (std::is_same_v<T, int>)
    {
        return object.value(key).toInt(fallback);
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        return object.value(key).toBool(fallback);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return static_cast<float>(object.value(key).toDouble(fallback));
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return object.value(key).toDouble(fallback);
    }
    else if constexpr (std::is_same_v<T, QString>)
    {
        return object.value(key).toString(fallback);
    }
}

} // namespace

AppConfig AppConfig::defaults()
{
    return {};
}

AppConfig AppConfig::fromJsonObject(const QJsonObject& object)
{
    AppConfig cfg = defaults();

    const auto geometryObj = object.value("GEOMETRY").toObject();
    cfg.geometry.volW = readValue(geometryObj, "VOL_W", cfg.geometry.volW);
    cfg.geometry.volH = readValue(geometryObj, "VOL_H", cfg.geometry.volH);
    cfg.geometry.totalAngles = readValue(geometryObj, "TOTAL_ANGLES", cfg.geometry.totalAngles);
    cfg.geometry.corShift = readValue(geometryObj, "COR_SHIFT", cfg.geometry.corShift);
    cfg.geometry.corAuto = readValue(geometryObj, "COR_AUTO", cfg.geometry.corAuto);
    cfg.geometry.angleOffsetDeg = readValue(geometryObj, "ANGLE_OFFSET_DEG", cfg.geometry.angleOffsetDeg);

    const auto phaseObj = object.value("PHASE").toObject();
    cfg.phase.enable = readValue(phaseObj, "ENABLE", cfg.phase.enable);
    cfg.phase.energyKev = readValue(phaseObj, "ENERGY_KEV", cfg.phase.energyKev);
    cfg.phase.distanceM = readValue(phaseObj, "DISTANCE_M", cfg.phase.distanceM);
    cfg.phase.pixelSizeUm = readValue(phaseObj, "PIXEL_SIZE_UM", cfg.phase.pixelSizeUm);
    cfg.phase.alphaRatio = readValue(phaseObj, "ALPHA_RATIO", cfg.phase.alphaRatio);

    const auto stripeObj = object.value("STRIPE").toObject();
    cfg.stripe.enable = readValue(stripeObj, "ENABLE", cfg.stripe.enable);
    cfg.stripe.method = readValue(stripeObj, "METHOD", cfg.stripe.method).toUpper();
    if (cfg.stripe.method.isEmpty())
    {
        cfg.stripe.method = QStringLiteral("FW");
    }
    cfg.stripe.level = std::clamp(readValue(stripeObj, "LEVEL", cfg.stripe.level), 0, 20);
    cfg.stripe.waveletName = readValue(stripeObj, "WNAME", cfg.stripe.waveletName);
    if (cfg.stripe.waveletName.isEmpty())
    {
        cfg.stripe.waveletName = QStringLiteral("db5");
    }
    cfg.stripe.sigma = std::max(0.01f, readValue(stripeObj, "SIGMA", cfg.stripe.sigma));
    cfg.stripe.pad = readValue(stripeObj, "PAD", cfg.stripe.pad);
    cfg.stripe.chunk = std::clamp(readValue(stripeObj, "NCHUNK", cfg.stripe.chunk), 1, 1024);
    cfg.stripe.enable = false;

    const auto preprocessObj = object.value("PREPROCESS").toObject();
    cfg.preprocess.method = PreprocessMethod::Log;
    cfg.preprocess.airValue = 1.0f;
    cfg.preprocess.taperWidth = readValue(preprocessObj, "TAPER_WIDTH", cfg.preprocess.taperWidth);
    cfg.preprocess.filterType = filterTypeFromString(
        readValue(preprocessObj, "FILTER_TYPE", toString(cfg.preprocess.filterType)));
    cfg.preprocess.flatGap = std::max(0, readValue(preprocessObj, "FLAT_GAP", cfg.preprocess.flatGap));
    cfg.preprocess.flatNum = std::max(1, readValue(preprocessObj, "FLAT_NUM", cfg.preprocess.flatNum));
    const QJsonArray flatGroups = preprocessObj.value("FLAT_GROUPS").toArray();
    for (const QJsonValue& value : flatGroups)
    {
        const QJsonObject obj = value.toObject();
        if (obj.isEmpty())
        {
            continue;
        }
        FlatGroupConfig group;
        group.label = obj.value("LABEL").toString();
        group.afterProjectionCount = obj.value("AFTER_PROJECTION_COUNT").toInt(group.afterProjectionCount);
        group.firstFlatIndex = obj.value("FIRST_FLAT_INDEX").toInt(group.firstFlatIndex);
        group.flatCount = obj.value("FLAT_COUNT").toInt(group.flatCount);
        if (group.flatCount > 0)
        {
            cfg.preprocess.flatGroups.append(group);
        }
    }

    const auto computeObj = object.value("COMPUTE").toObject();
    cfg.compute.batchSize = readValue(computeObj, "BATCH_SIZE", cfg.compute.batchSize);
    cfg.compute.zChunkSize = readValue(computeObj, "Z_CHUNK_SIZE", cfg.compute.zChunkSize);
    cfg.compute.gpuIndex = readValue(computeObj, "GPU_INDEX", cfg.compute.gpuIndex);
    cfg.compute.zBlockSplit = readValue(computeObj, "Z_BLOCK_SPLIT", cfg.compute.zBlockSplit);

    const auto previewObj = object.value("VIS").toObject();
    cfg.preview.enableLive = readValue(previewObj, "ENABLE_LIVE", cfg.preview.enableLive);
    cfg.preview.liveSliceIndex = readValue(previewObj, "LIVE_SLICE_INDEX", cfg.preview.liveSliceIndex);

    const auto ctFlyObj = object.value("CT_FLY").toObject();
    cfg.ctFly.allowedBlurPx = std::clamp(
        readValue(ctFlyObj, "ALLOWED_BLUR_PX", cfg.ctFly.allowedBlurPx),
        0.01,
        1000.0);

    const auto cameraObj = object.value("CAMERA").toObject();
    cfg.cameraUi.exposureValue = std::clamp(
        readValue(cameraObj, "EXPOSURE_VALUE", cfg.cameraUi.exposureValue),
        0.001,
        600000.0);
    cfg.cameraUi.exposureUnit = readValue(cameraObj, "EXPOSURE_UNIT", cfg.cameraUi.exposureUnit);
    if (cfg.cameraUi.exposureUnit != QStringLiteral("us")
        && cfg.cameraUi.exposureUnit != QStringLiteral("ms")
        && cfg.cameraUi.exposureUnit != QStringLiteral("s"))
    {
        cfg.cameraUi.exposureUnit = QStringLiteral("ms");
    }
    cfg.cameraUi.roiX = std::clamp(readValue(cameraObj, "ROI_X", cfg.cameraUi.roiX), 0, 3199);
    cfg.cameraUi.roiY = std::clamp(readValue(cameraObj, "ROI_Y", cfg.cameraUi.roiY), 0, 3199);
    cfg.cameraUi.roiWidth = std::clamp(readValue(cameraObj, "ROI_WIDTH", cfg.cameraUi.roiWidth), 1, 3200);
    cfg.cameraUi.roiHeight = std::clamp(readValue(cameraObj, "ROI_HEIGHT", cfg.cameraUi.roiHeight), 1, 3200);
    cfg.cameraUi.snapPrefix = readValue(cameraObj, "SNAP_PREFIX", cfg.cameraUi.snapPrefix);
    cfg.cameraUi.snapPadding = std::clamp(readValue(cameraObj, "SNAP_PADDING", cfg.cameraUi.snapPadding), 1, 12);
    cfg.cameraUi.snapIndex = std::clamp(readValue(cameraObj, "SNAP_INDEX", cfg.cameraUi.snapIndex), 0, 999999999);
    cfg.cameraUi.saveDir = readValue(cameraObj, "SAVE_DIR", cfg.cameraUi.saveDir);

    const auto ctAcqObj = object.value("CT_ACQUISITION").toObject();
    cfg.ctAcquisitionUi.beginningAngleDeg = std::clamp(
        readValue(ctAcqObj, "BEGINNING_ANGLE_DEG", cfg.ctAcquisitionUi.beginningAngleDeg),
        -360.0,
        360.0);
    cfg.ctAcquisitionUi.modeIndex = std::clamp(readValue(ctAcqObj, "MODE_INDEX", cfg.ctAcquisitionUi.modeIndex), 0, 2);
    cfg.ctAcquisitionUi.savePath = readValue(ctAcqObj, "SAVE_PATH", cfg.ctAcquisitionUi.savePath);
    cfg.ctAcquisitionUi.projectionNum = std::clamp(readValue(ctAcqObj, "PROJECTION_NUM", cfg.ctAcquisitionUi.projectionNum), 1, 20000);
    const int legacyFlatGap = std::clamp(readValue(ctAcqObj, "FLAT_GAP", cfg.ctAcquisitionUi.stepFlatGap), 0, 20000);
    const double legacyFlatMove = std::clamp(readValue(ctAcqObj, "FLAT_MOVE_UM", cfg.ctAcquisitionUi.stepFlatMoveUm), -500000.0, 500000.0);
    const int legacyFlatNum = std::clamp(readValue(ctAcqObj, "FLAT_NUM", cfg.ctAcquisitionUi.stepFlatNum), 0, 10000);
    cfg.ctAcquisitionUi.stepFlatGap = std::clamp(readValue(ctAcqObj, "STEP_FLAT_GAP", legacyFlatGap), 0, 20000);
    cfg.ctAcquisitionUi.stepFlatMoveUm = std::clamp(readValue(ctAcqObj, "STEP_FLAT_MOVE_UM", legacyFlatMove), -500000.0, 500000.0);
    cfg.ctAcquisitionUi.stepFlatNum = std::clamp(readValue(ctAcqObj, "STEP_FLAT_NUM", legacyFlatNum), 0, 10000);
    cfg.ctAcquisitionUi.flyFlatMoveUm = std::clamp(readValue(ctAcqObj, "FLY_FLAT_MOVE_UM", legacyFlatMove), -500000.0, 500000.0);
    cfg.ctAcquisitionUi.flyFlatNum = std::clamp(readValue(ctAcqObj, "FLY_FLAT_NUM", legacyFlatNum), 0, 10000);
    cfg.ctAcquisitionUi.flyLeadAngleDeg = std::clamp(readValue(ctAcqObj, "FLY_LEAD_ANGLE_DEG", cfg.ctAcquisitionUi.flyLeadAngleDeg), 0.0, 90.0);
    cfg.ctAcquisitionUi.rawFrameStride = std::clamp(readValue(ctAcqObj, "RAW_FRAME_STRIDE", cfg.ctAcquisitionUi.rawFrameStride), 1, 1000000);
    cfg.ctAcquisitionUi.tiffThreads = std::clamp(readValue(ctAcqObj, "TIFF_THREADS", cfg.ctAcquisitionUi.tiffThreads), 1, 16);

    const auto kohzuObj = object.value("KOHZU").toObject();
    cfg.kohzuUi.host = readValue(kohzuObj, "HOST", cfg.kohzuUi.host);
    cfg.kohzuUi.port = std::clamp(readValue(kohzuObj, "PORT", cfg.kohzuUi.port), 1, 65535);

    const auto ioObj = object.value("IO").toObject();
    cfg.io.watchDir = readValue(ioObj, "WATCH_DIR", cfg.io.watchDir);
    cfg.io.saveDir = readValue(ioObj, "SAVE_DIR", cfg.io.saveDir);
    cfg.io.reconMethod = reconMethodFromString(
        readValue(ioObj, "RECON_METHOD", toString(cfg.io.reconMethod)));
    cfg.io.correctionMode = correctionModeFromString(
        readValue(ioObj, "CORRECTION_MODE", toString(cfg.io.correctionMode)));
    cfg.io.tomoPrefix = readValue(ioObj, "TOMO_PREFIX", cfg.io.tomoPrefix);
    cfg.io.flatPrefix = readValue(ioObj, "FLAT_PREFIX", cfg.io.flatPrefix);
    cfg.io.darkPrefix = readValue(ioObj, "DARK_PREFIX", cfg.io.darkPrefix);
    cfg.io.inputPadding = readValue(ioObj, "INPUT_PADDING", cfg.io.inputPadding);
    cfg.io.inputExt = readValue(ioObj, "INPUT_EXT", cfg.io.inputExt);
    cfg.io.filePattern0 = readValue(ioObj, "FILE_PATTERN_0", cfg.io.filePattern0);
    cfg.io.filePattern180 = readValue(ioObj, "FILE_PATTERN_180", cfg.io.filePattern180);
    cfg.io.filePrefix = readValue(ioObj, "FILE_PREFIX", cfg.io.filePrefix);
    cfg.io.saveDepth = saveDepthFromString(readValue(ioObj, "SAVE_DEPTH", toString(cfg.io.saveDepth)));
    cfg.io.saveFormat = readValue(ioObj, "SAVE_FMT", cfg.io.saveFormat);
    cfg.io.tiffThreads = std::clamp(readValue(ioObj, "TIFF_THREADS", cfg.io.tiffThreads), 1, 64);

    const auto motorObj = object.value("MOTOR").toObject();
    for (auto it = motorObj.begin(); it != motorObj.end(); ++it)
    {
        const QJsonObject axisObj = it.value().toObject();
        if (axisObj.isEmpty())
        {
            continue;
        }
        MotorAxisConfig axisCfg;
        axisCfg.hasRelativeStep = axisObj.contains("RELATIVE_STEP");
        axisCfg.relativeStep = axisObj.value("RELATIVE_STEP").toDouble(axisCfg.relativeStep);
        axisCfg.hasAbsoluteTarget = axisObj.contains("ABSOLUTE_TARGET");
        axisCfg.absoluteTarget = axisObj.value("ABSOLUTE_TARGET").toDouble(axisCfg.absoluteTarget);
        axisCfg.hasBaseSpeed = axisObj.contains("START_SPEED_PER_S") || axisObj.contains("BASE_SPEED");
        axisCfg.baseSpeed = axisObj.contains("START_SPEED_PER_S")
            ? axisObj.value("START_SPEED_PER_S").toDouble(axisCfg.baseSpeed)
            : axisObj.value("BASE_SPEED").toDouble(axisCfg.baseSpeed);
        axisCfg.speed = axisObj.contains("TOP_SPEED_PER_S")
            ? axisObj.value("TOP_SPEED_PER_S").toDouble(axisCfg.speed)
            : axisObj.value("SPEED").toDouble(axisCfg.speed);
        axisCfg.accel = axisObj.contains("ACCEL_TIME_S")
            ? axisObj.value("ACCEL_TIME_S").toDouble(axisCfg.accel)
            : axisObj.value("ACCEL").toDouble(axisCfg.accel);
        cfg.motorAxes.insert(it.key(), axisCfg);
    }

    return cfg;
}

AppConfig AppConfig::loadFromJsonFile(const QString& filePath)
{
    AppConfig config = defaults();
    if (!tryLoadFromJsonFile(filePath, &config, nullptr))
    {
        return defaults();
    }

    return config;
}

bool AppConfig::tryLoadFromJsonFile(const QString& filePath, AppConfig* outConfig, QString* errorMessage)
{
    if (outConfig == nullptr)
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Output config pointer is null.");
        }
        return false;
    }

    QFile file(filePath);
    if (!file.exists())
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Config file does not exist: %1").arg(filePath);
        }
        return false;
    }

    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to open config file: %1").arg(filePath);
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError)
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Failed to parse config JSON: %1").arg(parseError.errorString());
        }
        return false;
    }

    if (!doc.isObject())
    {
        if (errorMessage != nullptr)
        {
            *errorMessage = QStringLiteral("Config JSON root is not an object.");
        }
        return false;
    }

    *outConfig = fromJsonObject(doc.object());
    return true;
}

QJsonObject AppConfig::toJsonObject() const
{
    QJsonObject root;

    QJsonObject geometryObj;
    geometryObj["VOL_W"] = geometry.volW;
    geometryObj["VOL_H"] = geometry.volH;
    geometryObj["TOTAL_ANGLES"] = geometry.totalAngles;
    geometryObj["COR_SHIFT"] = geometry.corShift;
    geometryObj["COR_AUTO"] = geometry.corAuto;
    geometryObj["ANGLE_OFFSET_DEG"] = geometry.angleOffsetDeg;
    root["GEOMETRY"] = geometryObj;

    QJsonObject phaseObj;
    phaseObj["ENABLE"] = phase.enable;
    phaseObj["ENERGY_KEV"] = phase.energyKev;
    phaseObj["DISTANCE_M"] = phase.distanceM;
    phaseObj["PIXEL_SIZE_UM"] = phase.pixelSizeUm;
    phaseObj["ALPHA_RATIO"] = phase.alphaRatio;
    root["PHASE"] = phaseObj;

    QJsonObject stripeObj;
    stripeObj["ENABLE"] = false;
    stripeObj["METHOD"] = stripe.method.isEmpty() ? QStringLiteral("FW") : stripe.method;
    stripeObj["LEVEL"] = stripe.level;
    stripeObj["WNAME"] = stripe.waveletName.isEmpty() ? QStringLiteral("db5") : stripe.waveletName;
    stripeObj["SIGMA"] = stripe.sigma;
    stripeObj["PAD"] = stripe.pad;
    stripeObj["NCHUNK"] = stripe.chunk;
    root["STRIPE"] = stripeObj;

    QJsonObject preprocessObj;
    preprocessObj["TAPER_WIDTH"] = preprocess.taperWidth;
    preprocessObj["FILTER_TYPE"] = toString(preprocess.filterType);
    preprocessObj["FLAT_GAP"] = preprocess.flatGap;
    preprocessObj["FLAT_NUM"] = preprocess.flatNum;
    if (!preprocess.flatGroups.isEmpty())
    {
        QJsonArray flatGroups;
        for (const FlatGroupConfig& group : preprocess.flatGroups)
        {
            QJsonObject obj;
            obj["LABEL"] = group.label;
            obj["AFTER_PROJECTION_COUNT"] = group.afterProjectionCount;
            obj["FIRST_FLAT_INDEX"] = group.firstFlatIndex;
            obj["FLAT_COUNT"] = group.flatCount;
            flatGroups.append(obj);
        }
        preprocessObj["FLAT_GROUPS"] = flatGroups;
    }
    root["PREPROCESS"] = preprocessObj;

    QJsonObject computeObj;
    computeObj["BATCH_SIZE"] = compute.batchSize;
    computeObj["Z_CHUNK_SIZE"] = compute.zChunkSize;
    computeObj["GPU_INDEX"] = compute.gpuIndex;
    computeObj["Z_BLOCK_SPLIT"] = compute.zBlockSplit;
    root["COMPUTE"] = computeObj;

    QJsonObject previewObj;
    previewObj["ENABLE_LIVE"] = preview.enableLive;
    previewObj["LIVE_SLICE_INDEX"] = preview.liveSliceIndex;
    root["VIS"] = previewObj;

    QJsonObject ctFlyObj;
    ctFlyObj["ALLOWED_BLUR_PX"] = ctFly.allowedBlurPx;
    root["CT_FLY"] = ctFlyObj;

    QJsonObject cameraObj;
    cameraObj["EXPOSURE_VALUE"] = cameraUi.exposureValue;
    cameraObj["EXPOSURE_UNIT"] = cameraUi.exposureUnit;
    cameraObj["ROI_X"] = cameraUi.roiX;
    cameraObj["ROI_Y"] = cameraUi.roiY;
    cameraObj["ROI_WIDTH"] = cameraUi.roiWidth;
    cameraObj["ROI_HEIGHT"] = cameraUi.roiHeight;
    cameraObj["SNAP_PREFIX"] = cameraUi.snapPrefix;
    cameraObj["SNAP_PADDING"] = cameraUi.snapPadding;
    cameraObj["SNAP_INDEX"] = cameraUi.snapIndex;
    cameraObj["SAVE_DIR"] = cameraUi.saveDir;
    root["CAMERA"] = cameraObj;

    QJsonObject ctAcqObj;
    ctAcqObj["BEGINNING_ANGLE_DEG"] = ctAcquisitionUi.beginningAngleDeg;
    ctAcqObj["MODE_INDEX"] = ctAcquisitionUi.modeIndex;
    ctAcqObj["SAVE_PATH"] = ctAcquisitionUi.savePath;
    ctAcqObj["PROJECTION_NUM"] = ctAcquisitionUi.projectionNum;
    ctAcqObj["STEP_FLAT_GAP"] = ctAcquisitionUi.stepFlatGap;
    ctAcqObj["STEP_FLAT_MOVE_UM"] = ctAcquisitionUi.stepFlatMoveUm;
    ctAcqObj["STEP_FLAT_NUM"] = ctAcquisitionUi.stepFlatNum;
    ctAcqObj["FLY_FLAT_MOVE_UM"] = ctAcquisitionUi.flyFlatMoveUm;
    ctAcqObj["FLY_FLAT_NUM"] = ctAcquisitionUi.flyFlatNum;
    ctAcqObj["FLY_LEAD_ANGLE_DEG"] = ctAcquisitionUi.flyLeadAngleDeg;
    ctAcqObj["RAW_FRAME_STRIDE"] = ctAcquisitionUi.rawFrameStride;
    ctAcqObj["TIFF_THREADS"] = ctAcquisitionUi.tiffThreads;
    root["CT_ACQUISITION"] = ctAcqObj;

    QJsonObject kohzuObj;
    kohzuObj["HOST"] = kohzuUi.host;
    kohzuObj["PORT"] = kohzuUi.port;
    root["KOHZU"] = kohzuObj;

    QJsonObject ioObj;
    ioObj["WATCH_DIR"] = io.watchDir;
    ioObj["SAVE_DIR"] = io.saveDir;
    ioObj["RECON_METHOD"] = toString(io.reconMethod);
    ioObj["CORRECTION_MODE"] = toString(io.correctionMode);
    ioObj["TOMO_PREFIX"] = io.tomoPrefix;
    ioObj["FLAT_PREFIX"] = io.flatPrefix;
    ioObj["DARK_PREFIX"] = io.darkPrefix;
    ioObj["INPUT_PADDING"] = io.inputPadding;
    ioObj["INPUT_EXT"] = io.inputExt;
    ioObj["FILE_PATTERN_0"] = io.filePattern0;
    ioObj["FILE_PATTERN_180"] = io.filePattern180;
    ioObj["FILE_PREFIX"] = io.filePrefix;
    ioObj["SAVE_DEPTH"] = toString(io.saveDepth);
    ioObj["SAVE_FMT"] = io.saveFormat;
    ioObj["TIFF_THREADS"] = io.tiffThreads;
    root["IO"] = ioObj;

    QJsonObject motorObj;
    for (auto it = motorAxes.begin(); it != motorAxes.end(); ++it)
    {
        QJsonObject axisObj;
        axisObj["RELATIVE_STEP"] = it.value().relativeStep;
        axisObj["ABSOLUTE_TARGET"] = it.value().absoluteTarget;
        axisObj["START_SPEED_PER_S"] = it.value().baseSpeed;
        axisObj["TOP_SPEED_PER_S"] = it.value().speed;
        axisObj["ACCEL_TIME_S"] = it.value().accel;
        axisObj["BASE_SPEED"] = it.value().baseSpeed;
        axisObj["SPEED"] = it.value().speed;
        axisObj["ACCEL"] = it.value().accel;
        motorObj[it.key()] = axisObj;
    }
    root["MOTOR"] = motorObj;

    return root;
}

bool AppConfig::saveToJsonFile(const QString& filePath) const
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }

    const QJsonDocument doc(toJsonObject());
    file.write(doc.toJson(QJsonDocument::Indented));
    return true;
}

QString toString(CorrectionMode mode)
{
    switch (mode)
    {
    case CorrectionMode::None: return "None";
    case CorrectionMode::Flat: return "Flat";
    case CorrectionMode::FlatDark: return "Flat-Dark";
    }
    return "Flat-Dark";
}

QString toString(PreprocessMethod method)
{
    switch (method)
    {
    case PreprocessMethod::Linear: return "linear";
    case PreprocessMethod::Log: return "log";
    }
    return "log";
}

QString toString(FilterType type)
{
    switch (type)
    {
    case FilterType::RamLak: return "ram-lak";
    case FilterType::SheppLogan: return "shepp-logan";
    case FilterType::Cosine: return "cosine";
    case FilterType::Hamming: return "hamming";
    case FilterType::Hann: return "hann";
    }
    return "shepp-logan";
}

QString toString(SaveDepth depth)
{
    switch (depth)
    {
    case SaveDepth::Bit8: return "8bit";
    case SaveDepth::Bit16: return "16bit";
    case SaveDepth::Bit32: return "32bit";
    }
    return "16bit";
}

QString toString(ReconMethod method)
{
    switch (method)
    {
    case ReconMethod::RealTimeFBP: return "RealTimeFBP";
    case ReconMethod::OfflineFBP: return "OfflineFBP";
    case ReconMethod::OfflineGridrec: return "OfflineGridrec";
    }
    return "RealTimeFBP";
}

CorrectionMode correctionModeFromString(const QString& value)
{
    if (value.compare("None", Qt::CaseInsensitive) == 0)
    {
        return CorrectionMode::None;
    }
    if (value.compare("Flat", Qt::CaseInsensitive) == 0)
    {
        return CorrectionMode::Flat;
    }
    return CorrectionMode::FlatDark;
}

PreprocessMethod preprocessMethodFromString(const QString& value)
{
    (void)value;
    return PreprocessMethod::Log;
}

FilterType filterTypeFromString(const QString& value)
{
    if (value.compare("ram-lak", Qt::CaseInsensitive) == 0) return FilterType::RamLak;
    if (value.compare("cosine", Qt::CaseInsensitive) == 0) return FilterType::Cosine;
    if (value.compare("hamming", Qt::CaseInsensitive) == 0) return FilterType::Hamming;
    if (value.compare("hann", Qt::CaseInsensitive) == 0) return FilterType::Hann;
    return FilterType::SheppLogan;
}

SaveDepth saveDepthFromString(const QString& value)
{
    if (value.compare("8bit", Qt::CaseInsensitive) == 0) return SaveDepth::Bit8;
    if (value.compare("32bit", Qt::CaseInsensitive) == 0) return SaveDepth::Bit32;
    return SaveDepth::Bit16;
}

ReconMethod reconMethodFromString(const QString& value)
{
    if (value.compare("OfflineFBP", Qt::CaseInsensitive) == 0
        || value.compare("offline-fbp", Qt::CaseInsensitive) == 0
        || value.compare("FBP Tomo", Qt::CaseInsensitive) == 0)
    {
        return ReconMethod::OfflineFBP;
    }
    if (value.compare("OfflineGridrec", Qt::CaseInsensitive) == 0
        || value.compare("offline-gridrec", Qt::CaseInsensitive) == 0
        || value.compare("Gridrec Tomo", Qt::CaseInsensitive) == 0)
    {
        return ReconMethod::OfflineGridrec;
    }
    return ReconMethod::RealTimeFBP;
}

} // namespace ct
