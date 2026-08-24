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
    qint64 exposureTimeNs = 0;
};

struct TimingAnalysis
{
    bool valid = false;
    bool stable = false;
    QString error;
    QString warning;
    int sampleCount = 0;
    qint64 firstFrameNumber = 0;
    qint64 lastFrameNumber = 0;
    qint64 droppedFrames = 0;
    double meanPeriodMs = 0.0;
    double medianPeriodMs = 0.0;
    double fittedPeriodMs = 0.0;
    double minPeriodMs = 0.0;
    double maxPeriodMs = 0.0;
    double standardDeviationMs = 0.0;
    double maxPeriodDeviationMs = 0.0;
    double firstHalfMeanPeriodMs = 0.0;
    double secondHalfMeanPeriodMs = 0.0;
    double meanDriftMs = 0.0;
    double meanDriftPercent = 0.0;
    double maxFitResidualMs = 0.0;
    bool exposureTimingAvailable = false;
    double meanExposureMs = 0.0;
    double minExposureMs = 0.0;
    double maxExposureMs = 0.0;
    double maxExposureDeviationMs = 0.0;
    double rawFrameRateFps = 0.0;
};

enum class ContinuousTimingObservation
{
    Accepted,
    Duplicate,
    DeviceFrameLoss,
    InvalidTimestamp,
    KeptFrameStrideViolation
};

struct ContinuousTimingStats
{
    bool valid = false;
    double baselinePeriodMs = 0.0;
    double overallPeriodMs = 0.0;
    double overallFrameRateFps = 0.0;
    double rollingPeriodMs = 0.0;
    double rollingFrameRateFps = 0.0;
    double rollingDriftPercent = 0.0;
    double currentIntervalMs = 0.0;
    double minIntervalMs = 0.0;
    double maxIntervalMs = 0.0;
    double maxIntervalDeviationMs = 0.0;
    double currentPhaseErrorMs = 0.0;
    double maxAbsPhaseErrorMs = 0.0;
    double fittedPeriodMs = 0.0;
    double firstHalfMeanPeriodMs = 0.0;
    double secondHalfMeanPeriodMs = 0.0;
    double meanDriftMs = 0.0;
    double meanDriftPercent = 0.0;
    double maxFitResidualMs = 0.0;
    qint64 rawFrames = 0;
    qint64 keptFrames = 0;
    qint64 duplicateNotifications = 0;
    qint64 deviceDroppedFrames = 0;
    bool frameLossDetected = false;
    bool invalidTimestampDetected = false;
    bool keptStrideViolationDetected = false;
    qint64 previousDeviceFrame = 0;
    qint64 currentDeviceFrame = 0;
    qint64 missingDeviceFrames = 0;
};

class ContinuousTimingTracker
{
public:
    void reset(double baselinePeriodMs, int keptFrameStride, int rollingWindowIntervals = 16);
    ContinuousTimingObservation observeRaw(qint64 deviceFrameNumber, qint64 bofTimestampNs);
    ContinuousTimingObservation markKept(qint64 deviceFrameNumber, qint64 bofTimestampNs);
    ContinuousTimingStats snapshot(bool includeFinalAnalysis = false) const;

private:
    double baselinePeriodMs_ = 0.0;
    int keptFrameStride_ = 1;
    int rollingWindowIntervals_ = 16;
    QVector<TimingSample> rawSamples_;
    QVector<double> periodsMs_;
    QVector<double> rollingPeriodsMs_;
    qint64 duplicateNotifications_ = 0;
    qint64 deviceDroppedFrames_ = 0;
    bool frameLossDetected_ = false;
    bool invalidTimestampDetected_ = false;
    bool keptStrideViolationDetected_ = false;
    qint64 lossPreviousFrame_ = 0;
    qint64 lossCurrentFrame_ = 0;
    qint64 lossMissingFrames_ = 0;
    qint64 firstKeptFrame_ = 0;
    qint64 previousKeptFrame_ = 0;
    qint64 firstKeptTimestampNs_ = 0;
    qint64 keptFrames_ = 0;
    double currentPhaseErrorMs_ = 0.0;
    double maxAbsPhaseErrorMs_ = 0.0;
    double minIntervalMs_ = 0.0;
    double maxIntervalMs_ = 0.0;
    double maxIntervalDeviationMs_ = 0.0;
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
    int endpointReferenceLogicalIndex = 0;
    qint64 endpointReferenceRawFrameOffset = 0;
    double endpointReferenceTimeMs = 0.0;
};

struct TimeSeriesTimingPlan
{
    bool valid = false;
    QString error;
    int rawFrameStride = 1;
    double actualIntervalMs = 0.0;
    double actualFrameRateFps = 0.0;
};

TimingAnalysis analyzeTimingSamples(const QVector<TimingSample>& samples);

FlyTimingPlan makeFlyTimingPlan(double totalAngleDeg, int projectionCount, double allowedBlurPx,
    double roiRadiusPx, double exposureMs, double rawFramePeriodMs);

TimeSeriesTimingPlan makeTimeSeriesTimingPlan(int rawFrameStride, double rawFramePeriodMs);

} // namespace ct
