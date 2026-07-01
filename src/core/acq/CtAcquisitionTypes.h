#pragma once

#include "core/config/AppConfig.h"
#include "hardware/camera/PvcamCameraWorker.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QVector>

#include <opencv2/core.hpp>

namespace ct
{

enum class CtAcquisitionMode
{
    StepAndShoot,
    FlyScan
};

struct CtAcquisitionSettings
{
    CtAcquisitionMode mode = CtAcquisitionMode::StepAndShoot;
    QString savePath;
    double beginningAngleDeg = 0.0;
    int projectionCount = 1080;
    int flatGap = 0;
    double flatMoveDistanceUm = 0.0;
    int flatCount = 10;
    double flySpeedDegPerSec = 1.0;
    double flyLeadAngleDeg = 2.0;
    double flyFrameIntervalMs = 0.0;
    double flyScanTimeS = 0.0;
    bool autoCor = false;
    int padding = 4;
    QString tomoPrefix = QStringLiteral("Tomo_");
    QString flatPrefix = QStringLiteral("Flat_");
    QString extension = QStringLiteral("tiff");
    int tiffThreads = 16;
    PvcamCaptureSettings camera;
    AppConfig reconConfig;
    double rPulsePerDeg = 1.0;
    double yPulsePerUm = 1.0;
    int rAxis = 17;
    int yAxis = 3;
};

struct CtFrameSaveItem
{
    QString filePath;
    cv::Mat image;
};

struct CtFlatGroup
{
    QString label;
    int afterProjectionCount = 0;
    int firstFlatIndex = 0;
    int flatCount = 0;
    double yMoveUm = 0.0;
};

inline QJsonObject flatGroupToJson(const CtFlatGroup& group)
{
    QJsonObject obj;
    obj["LABEL"] = group.label;
    obj["AFTER_PROJECTION_COUNT"] = group.afterProjectionCount;
    obj["FIRST_FLAT_INDEX"] = group.firstFlatIndex;
    obj["FLAT_COUNT"] = group.flatCount;
    obj["Y_MOVE_UM"] = group.yMoveUm;
    return obj;
}

inline QJsonArray flatGroupsToJson(const QVector<CtFlatGroup>& groups)
{
    QJsonArray array;
    for (const CtFlatGroup& group : groups)
    {
        array.append(flatGroupToJson(group));
    }
    return array;
}

} // namespace ct
