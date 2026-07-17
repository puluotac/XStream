#pragma once

#include <QVector>
#include <QString>

#include <QtGlobal>

namespace ct
{

struct TimingSample
{
    qint64 frameNumber = 0;
    qint64 timestampNs = 0;
};

struct TimingAnalysis
{
    bool valid = false;
    bool stable = false;
    QString error;
    int sampleCount = 0;
    qint64 firstFrameNumber = 0;
    qint64 lastFrameNumber = 0;
    qint64 droppedFrames = 0;
    double meanPeriodMs = 0.0;
    double medianPeriodMs = 0.0;
    double minPeriodMs = 0.0;
    double maxPeriodMs = 0.0;
    double standardDeviationMs = 0.0;
    double p95DeviationMs = 0.0;
    double rawFrameRateFps = 0.0;
};

struct FlyTimingPlan
{
    bool valid = false;
    QString error;
    int rawFrameStride = 1;
    double blurAngleDeg = 0.0;
    double maxFlySpeedDegPerSec = 0.0;
    double angleStepDeg = 0.0;
    double projectionIntervalMs = 0.0;
    double flySpeedDegPerSec = 0.0;
    double scanTimeS = 0.0;
};

struct TimeSeriesTimingPlan
{
    bool valid = false;
    QString error;
    int rawFrameStride = 1;
    double requestedIntervalMs = 0.0;
    double actualIntervalMs = 0.0;
    double actualFrameRateFps = 0.0;
};

TimingAnalysis analyzeTimingSamples(const QVector<TimingSample>& samples);

FlyTimingPlan makeFlyTimingPlan(double totalAngleDeg, int projectionCount, double allowedBlurPx,
    double roiRadiusPx, double exposureMs, double rawFramePeriodMs);

TimeSeriesTimingPlan makeTimeSeriesTimingPlan(double requestedIntervalMs, double rawFramePeriodMs);

} // namespace ct
