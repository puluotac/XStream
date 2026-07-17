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

    result.meanPeriodMs = std::accumulate(periodsMs.cbegin(), periodsMs.cend(), 0.0)
        / static_cast<double>(periodsMs.size());
    result.medianPeriodMs = percentile(periodsMs, 0.5);
    const auto minMax = std::minmax_element(periodsMs.cbegin(), periodsMs.cend());
    result.minPeriodMs = *minMax.first;
    result.maxPeriodMs = *minMax.second;

    double squaredSum = 0.0;
    QVector<double> deviations;
    deviations.reserve(periodsMs.size());
    for (double periodMs : periodsMs)
    {
        const double meanDelta = periodMs - result.meanPeriodMs;
        squaredSum += meanDelta * meanDelta;
        deviations.append(std::abs(periodMs - result.medianPeriodMs));
    }
    result.standardDeviationMs = std::sqrt(squaredSum / static_cast<double>(periodsMs.size()));
    result.p95DeviationMs = percentile(deviations, 0.95);
    result.rawFrameRateFps = result.medianPeriodMs > 0.0 ? 1000.0 / result.medianPeriodMs : 0.0;
    result.valid = std::isfinite(result.medianPeriodMs) && result.medianPeriodMs > 0.0;
    if (!result.valid)
    {
        result.error = QStringLiteral("Measured frame period is invalid.");
        return result;
    }

    const double allowedDeviationMs = std::max(0.25, result.medianPeriodMs * 0.02);
    result.stable = result.droppedFrames == 0 && result.p95DeviationMs <= allowedDeviationMs;
    if (!result.stable)
    {
        result.error = result.droppedFrames > 0
            ? QStringLiteral("Device frame sequence contains %1 missing frame(s).").arg(result.droppedFrames)
            : QStringLiteral("Frame-period p95 deviation %1 ms exceeds the %2 ms limit.")
                .arg(result.p95DeviationMs, 0, 'f', 4)
                .arg(allowedDeviationMs, 0, 'f', 4);
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
