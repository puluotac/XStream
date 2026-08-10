#include "core/acq/ContinuousTiming.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ct
{

namespace
{

double percentile(QVector<double> values, double fraction)
{
    if (values.isEmpty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = std::clamp(fraction, 0.0, 1.0);
    const double position = clamped * static_cast<double>(values.size() - 1);
    const int lower = static_cast<int>(std::floor(position));
    const int upper = static_cast<int>(std::ceil(position));
    if (lower == upper)
    {
        return values[lower];
    }
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

} // namespace

TimingAnalysis analyzeTimingSamples(const QVector<TimingSample>& samples)
{
    TimingAnalysis result;
    result.sampleCount = samples.size();
    if (samples.size() < 3)
    {
        result.error = QStringLiteral("At least three timing samples are required.");
        return result;
    }

    result.firstFrameNumber = samples.first().frameNumber;
    result.lastFrameNumber = samples.last().frameNumber;
    QVector<double> periodsMs;
    periodsMs.reserve(samples.size() - 1);
    QVector<double> exposureTimesMs;
    exposureTimesMs.reserve(samples.size());

    for (int i = 1; i < samples.size(); ++i)
    {
        const qint64 frameDelta = samples[i].frameNumber - samples[i - 1].frameNumber;
        const qint64 timeDeltaNs = samples[i].timestampNs - samples[i - 1].timestampNs;
        if (frameDelta <= 0)
        {
            result.error = QStringLiteral("Device frame numbers are not strictly increasing.");
            return result;
        }
        if (timeDeltaNs <= 0)
        {
            result.error = QStringLiteral("Frame timestamps are not strictly increasing.");
            return result;
        }
        result.droppedFrames += std::max<qint64>(0, frameDelta - 1);
        periodsMs.append(static_cast<double>(timeDeltaNs) / 1000000.0 / static_cast<double>(frameDelta));
    }
    for (const TimingSample& sample : samples)
    {
        if (sample.exposureTimeNs > 0)
        {
            exposureTimesMs.append(static_cast<double>(sample.exposureTimeNs) / 1000000.0);
        }
    }

    result.meanPeriodMs = std::accumulate(periodsMs.cbegin(), periodsMs.cend(), 0.0)
        / static_cast<double>(periodsMs.size());
    result.medianPeriodMs = percentile(periodsMs, 0.5);
    const auto minMax = std::minmax_element(periodsMs.cbegin(), periodsMs.cend());
    result.minPeriodMs = *minMax.first;
    result.maxPeriodMs = *minMax.second;

    const double firstFrame = static_cast<double>(samples.first().frameNumber);
    const double firstTimestampMs = static_cast<double>(samples.first().timestampNs) / 1000000.0;
    double xSum = 0.0;
    double ySum = 0.0;
    double xxSum = 0.0;
    double xySum = 0.0;
    for (const TimingSample& sample : samples)
    {
        const double x = static_cast<double>(sample.frameNumber) - firstFrame;
        const double y = static_cast<double>(sample.timestampNs) / 1000000.0 - firstTimestampMs;
        xSum += x;
        ySum += y;
        xxSum += x * x;
        xySum += x * y;
    }
    const double count = static_cast<double>(samples.size());
    const double denominator = count * xxSum - xSum * xSum;
    if (!(denominator > 0.0))
    {
        result.error = QStringLiteral("Unable to fit the measured frame period.");
        return result;
    }
    result.fittedPeriodMs = (count * xySum - xSum * ySum) / denominator;
    const double interceptMs = (ySum - result.fittedPeriodMs * xSum) / count;

    double squaredSum = 0.0;
    for (double periodMs : periodsMs)
    {
        const double meanDelta = periodMs - result.meanPeriodMs;
        squaredSum += meanDelta * meanDelta;
        result.maxPeriodDeviationMs = std::max(result.maxPeriodDeviationMs,
            std::abs(periodMs - result.fittedPeriodMs));
    }
    result.standardDeviationMs = std::sqrt(squaredSum / static_cast<double>(periodsMs.size()));
    for (const TimingSample& sample : samples)
    {
        const double x = static_cast<double>(sample.frameNumber) - firstFrame;
        const double y = static_cast<double>(sample.timestampNs) / 1000000.0 - firstTimestampMs;
        result.maxFitResidualMs = std::max(result.maxFitResidualMs,
            std::abs(y - (interceptMs + result.fittedPeriodMs * x)));
    }

    const int half = periodsMs.size() / 2;
    if (half > 0)
    {
        result.firstHalfMeanPeriodMs = std::accumulate(periodsMs.cbegin(), periodsMs.cbegin() + half, 0.0)
            / static_cast<double>(half);
        result.secondHalfMeanPeriodMs = std::accumulate(periodsMs.cbegin() + half, periodsMs.cend(), 0.0)
            / static_cast<double>(periodsMs.size() - half);
        result.meanDriftMs = result.secondHalfMeanPeriodMs - result.firstHalfMeanPeriodMs;
        result.meanDriftPercent = result.fittedPeriodMs > 0.0
            ? result.meanDriftMs / result.fittedPeriodMs * 100.0
            : 0.0;
    }

    if (!exposureTimesMs.isEmpty())
    {
        result.exposureTimingAvailable = exposureTimesMs.size() == samples.size();
        result.meanExposureMs = std::accumulate(exposureTimesMs.cbegin(), exposureTimesMs.cend(), 0.0)
            / static_cast<double>(exposureTimesMs.size());
        const auto exposureMinMax = std::minmax_element(exposureTimesMs.cbegin(), exposureTimesMs.cend());
        result.minExposureMs = *exposureMinMax.first;
        result.maxExposureMs = *exposureMinMax.second;
        for (double exposureMs : exposureTimesMs)
        {
            result.maxExposureDeviationMs = std::max(result.maxExposureDeviationMs,
                std::abs(exposureMs - result.meanExposureMs));
        }
    }

    result.rawFrameRateFps = result.fittedPeriodMs > 0.0 ? 1000.0 / result.fittedPeriodMs : 0.0;
    result.valid = std::isfinite(result.fittedPeriodMs) && result.fittedPeriodMs > 0.0;
    if (!result.valid)
    {
        result.error = QStringLiteral("Measured frame period is invalid.");
        return result;
    }

    const double warningDeviationMs = std::max(0.1, result.fittedPeriodMs * 0.01);
    result.stable = result.droppedFrames == 0 && result.maxPeriodDeviationMs <= warningDeviationMs;
    if (!result.stable)
    {
        result.warning = result.droppedFrames > 0
            ? QStringLiteral("Device frame sequence contains %1 missing frame(s).").arg(result.droppedFrames)
            : QStringLiteral("Maximum frame-period deviation %1 ms exceeds the %2 ms warning level.")
                .arg(result.maxPeriodDeviationMs, 0, 'f', 4)
                .arg(warningDeviationMs, 0, 'f', 4);
    }
    return result;
}

void ContinuousTimingTracker::reset(double baselinePeriodMs, int keptFrameStride, int rollingWindowIntervals)
{
    baselinePeriodMs_ = std::max(0.0, baselinePeriodMs);
    keptFrameStride_ = std::max(1, keptFrameStride);
    rollingWindowIntervals_ = std::max(1, rollingWindowIntervals);
    rawSamples_.clear();
    periodsMs_.clear();
    rollingPeriodsMs_.clear();
    duplicateNotifications_ = 0;
    deviceDroppedFrames_ = 0;
    frameLossDetected_ = false;
    invalidTimestampDetected_ = false;
    keptStrideViolationDetected_ = false;
    lossPreviousFrame_ = 0;
    lossCurrentFrame_ = 0;
    lossMissingFrames_ = 0;
    firstKeptFrame_ = 0;
    previousKeptFrame_ = 0;
    firstKeptTimestampNs_ = 0;
    keptFrames_ = 0;
    currentPhaseErrorMs_ = 0.0;
    maxAbsPhaseErrorMs_ = 0.0;
    minIntervalMs_ = 0.0;
    maxIntervalMs_ = 0.0;
    maxIntervalDeviationMs_ = 0.0;
}

ContinuousTimingObservation ContinuousTimingTracker::observeRaw(
    qint64 deviceFrameNumber, qint64 bofTimestampNs)
{
    if (bofTimestampNs <= 0)
    {
        invalidTimestampDetected_ = true;
        lossCurrentFrame_ = deviceFrameNumber;
        return ContinuousTimingObservation::InvalidTimestamp;
    }
    if (!rawSamples_.isEmpty())
    {
        const TimingSample& previous = rawSamples_.last();
        if (deviceFrameNumber == previous.frameNumber)
        {
            ++duplicateNotifications_;
            return ContinuousTimingObservation::Duplicate;
        }
        if (deviceFrameNumber != previous.frameNumber + 1)
        {
            frameLossDetected_ = true;
            lossPreviousFrame_ = previous.frameNumber;
            lossCurrentFrame_ = deviceFrameNumber;
            lossMissingFrames_ = deviceFrameNumber > previous.frameNumber + 1
                ? deviceFrameNumber - previous.frameNumber - 1
                : 0;
            deviceDroppedFrames_ += lossMissingFrames_;
            return ContinuousTimingObservation::DeviceFrameLoss;
        }
        if (bofTimestampNs <= previous.timestampNs)
        {
            invalidTimestampDetected_ = true;
            lossPreviousFrame_ = previous.frameNumber;
            lossCurrentFrame_ = deviceFrameNumber;
            return ContinuousTimingObservation::InvalidTimestamp;
        }

        const double periodMs = static_cast<double>(bofTimestampNs - previous.timestampNs) / 1000000.0;
        periodsMs_.append(periodMs);
        rollingPeriodsMs_.append(periodMs);
        while (rollingPeriodsMs_.size() > rollingWindowIntervals_)
        {
            rollingPeriodsMs_.removeFirst();
        }
        if (periodsMs_.size() == 1)
        {
            minIntervalMs_ = periodMs;
            maxIntervalMs_ = periodMs;
        }
        else
        {
            minIntervalMs_ = std::min(minIntervalMs_, periodMs);
            maxIntervalMs_ = std::max(maxIntervalMs_, periodMs);
        }
        if (baselinePeriodMs_ > 0.0)
        {
            maxIntervalDeviationMs_ = std::max(maxIntervalDeviationMs_,
                std::abs(periodMs - baselinePeriodMs_));
        }
    }

    rawSamples_.append({ deviceFrameNumber, bofTimestampNs, 0 });
    return ContinuousTimingObservation::Accepted;
}

ContinuousTimingObservation ContinuousTimingTracker::markKept(
    qint64 deviceFrameNumber, qint64 bofTimestampNs)
{
    if (keptFrames_ == 0)
    {
        firstKeptFrame_ = deviceFrameNumber;
        previousKeptFrame_ = deviceFrameNumber;
        firstKeptTimestampNs_ = bofTimestampNs;
    }
    else
    {
        if (deviceFrameNumber != previousKeptFrame_ + keptFrameStride_)
        {
            keptStrideViolationDetected_ = true;
            lossPreviousFrame_ = previousKeptFrame_;
            lossCurrentFrame_ = deviceFrameNumber;
            return ContinuousTimingObservation::KeptFrameStrideViolation;
        }
        const double actualElapsedMs = static_cast<double>(bofTimestampNs - firstKeptTimestampNs_) / 1000000.0;
        const double expectedElapsedMs = static_cast<double>(keptFrames_)
            * static_cast<double>(keptFrameStride_) * baselinePeriodMs_;
        currentPhaseErrorMs_ = actualElapsedMs - expectedElapsedMs;
        maxAbsPhaseErrorMs_ = std::max(maxAbsPhaseErrorMs_, std::abs(currentPhaseErrorMs_));
        previousKeptFrame_ = deviceFrameNumber;
    }
    ++keptFrames_;
    return ContinuousTimingObservation::Accepted;
}

ContinuousTimingStats ContinuousTimingTracker::snapshot(bool includeFinalAnalysis) const
{
    ContinuousTimingStats result;
    result.baselinePeriodMs = baselinePeriodMs_;
    result.rawFrames = rawSamples_.size();
    result.keptFrames = keptFrames_;
    result.duplicateNotifications = duplicateNotifications_;
    result.deviceDroppedFrames = deviceDroppedFrames_;
    result.frameLossDetected = frameLossDetected_;
    result.invalidTimestampDetected = invalidTimestampDetected_;
    result.keptStrideViolationDetected = keptStrideViolationDetected_;
    result.previousDeviceFrame = lossPreviousFrame_;
    result.currentDeviceFrame = lossCurrentFrame_;
    result.missingDeviceFrames = lossMissingFrames_;
    result.currentPhaseErrorMs = currentPhaseErrorMs_;
    result.maxAbsPhaseErrorMs = maxAbsPhaseErrorMs_;
    result.minIntervalMs = minIntervalMs_;
    result.maxIntervalMs = maxIntervalMs_;
    result.maxIntervalDeviationMs = maxIntervalDeviationMs_;
    if (rawSamples_.size() >= 2)
    {
        const TimingSample& first = rawSamples_.first();
        const TimingSample& last = rawSamples_.last();
        const qint64 frameDelta = last.frameNumber - first.frameNumber;
        if (frameDelta > 0)
        {
            result.overallPeriodMs = static_cast<double>(last.timestampNs - first.timestampNs)
                / 1000000.0 / static_cast<double>(frameDelta);
            result.overallFrameRateFps = result.overallPeriodMs > 0.0
                ? 1000.0 / result.overallPeriodMs
                : 0.0;
        }
        result.currentIntervalMs = periodsMs_.isEmpty() ? 0.0 : periodsMs_.last();
        if (!rollingPeriodsMs_.isEmpty())
        {
            result.rollingPeriodMs = std::accumulate(rollingPeriodsMs_.cbegin(), rollingPeriodsMs_.cend(), 0.0)
                / static_cast<double>(rollingPeriodsMs_.size());
            result.rollingFrameRateFps = result.rollingPeriodMs > 0.0
                ? 1000.0 / result.rollingPeriodMs
                : 0.0;
            result.rollingDriftPercent = baselinePeriodMs_ > 0.0
                ? (result.rollingPeriodMs - baselinePeriodMs_) / baselinePeriodMs_ * 100.0
                : 0.0;
        }
        result.valid = result.overallPeriodMs > 0.0;
    }
    if (includeFinalAnalysis && rawSamples_.size() >= 3)
    {
        const TimingAnalysis analysis = analyzeTimingSamples(rawSamples_);
        if (analysis.valid)
        {
            result.fittedPeriodMs = analysis.fittedPeriodMs;
            result.firstHalfMeanPeriodMs = analysis.firstHalfMeanPeriodMs;
            result.secondHalfMeanPeriodMs = analysis.secondHalfMeanPeriodMs;
            result.meanDriftMs = analysis.meanDriftMs;
            result.meanDriftPercent = analysis.meanDriftPercent;
            result.maxFitResidualMs = analysis.maxFitResidualMs;
        }
    }
    return result;
}

FlyTimingPlan makeFlyTimingPlan(double totalAngleDeg, int projectionCount, double allowedBlurPx,
    double roiRadiusPx, double exposureMs, double rawFramePeriodMs)
{
    FlyTimingPlan plan;
    if (!(totalAngleDeg > 0.0) || projectionCount <= 0 || !(allowedBlurPx > 0.0)
        || !(roiRadiusPx > 0.0) || !(exposureMs > 0.0) || !(rawFramePeriodMs > 0.0))
    {
        plan.error = QStringLiteral("Fly timing inputs must all be positive.");
        return plan;
    }

    const double radiansToDegrees = 180.0 / std::acos(-1.0);
    plan.blurAngleDeg = std::atan(allowedBlurPx / roiRadiusPx) * radiansToDegrees;
    plan.maxFlySpeedDegPerSec = plan.blurAngleDeg / (exposureMs / 1000.0);
    plan.angleStepDeg = totalAngleDeg / static_cast<double>(projectionCount);
    if (!(plan.maxFlySpeedDegPerSec > 0.0) || !std::isfinite(plan.maxFlySpeedDegPerSec))
    {
        plan.error = QStringLiteral("The blur-limited Fly speed is invalid.");
        return plan;
    }

    const double rawPeriodS = rawFramePeriodMs / 1000.0;
    const double minimumStride = plan.angleStepDeg / (plan.maxFlySpeedDegPerSec * rawPeriodS);
    plan.rawFrameStride = std::max(1, static_cast<int>(std::ceil(minimumStride - 1e-12)));
    plan.projectionIntervalMs = rawFramePeriodMs * static_cast<double>(plan.rawFrameStride);
    plan.flySpeedDegPerSec = plan.angleStepDeg / (plan.projectionIntervalMs / 1000.0);
    plan.scanTimeS = static_cast<double>(projectionCount) * plan.projectionIntervalMs / 1000.0;
    plan.valid = std::isfinite(plan.flySpeedDegPerSec) && plan.flySpeedDegPerSec > 0.0
        && plan.flySpeedDegPerSec <= plan.maxFlySpeedDegPerSec * (1.0 + 1e-9);
    if (!plan.valid)
    {
        plan.error = QStringLiteral("Unable to derive a valid integer-stride Fly plan.");
    }
    return plan;
}

TimeSeriesTimingPlan makeTimeSeriesTimingPlan(double requestedIntervalMs, double rawFramePeriodMs)
{
    TimeSeriesTimingPlan plan;
    plan.requestedIntervalMs = requestedIntervalMs;
    if (!(requestedIntervalMs > 0.0) || !(rawFramePeriodMs > 0.0))
    {
        plan.error = QStringLiteral("2D timing inputs must be positive.");
        return plan;
    }

    plan.rawFrameStride = std::max(1,
        static_cast<int>(std::ceil(requestedIntervalMs / rawFramePeriodMs - 1e-12)));
    plan.actualIntervalMs = rawFramePeriodMs * static_cast<double>(plan.rawFrameStride);
    plan.actualFrameRateFps = 1000.0 / plan.actualIntervalMs;
    plan.valid = std::isfinite(plan.actualFrameRateFps) && plan.actualFrameRateFps > 0.0;
    if (!plan.valid)
    {
        plan.error = QStringLiteral("Unable to derive a valid integer-stride 2D plan.");
    }
    return plan;
}

} // namespace ct
