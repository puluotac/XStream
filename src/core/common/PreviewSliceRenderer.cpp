#include "core/common/PreviewSliceRenderer.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ct
{

namespace
{

constexpr double kLowerPercentile = 1.0;
constexpr double kUpperPercentile = 99.0;
constexpr int kMaxSamples = 65536;

PreviewWindow minMaxWindow(const cv::Mat& slice)
{
    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(slice, &minValue, &maxValue);

    if (maxValue <= minValue)
    {
        maxValue = minValue + 1.0;
    }

    return { minValue, maxValue };
}

void appendFiniteSamples(
    const cv::Mat& slice,
    size_t targetSampleCount,
    std::vector<float>& outSamples)
{
    if (slice.empty() || targetSampleCount == 0)
    {
        return;
    }

    const int total = slice.rows * slice.cols;
    if (total <= 0)
    {
        return;
    }

    const int sampleStride = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(total) / targetSampleCount)));
    outSamples.reserve(outSamples.size() + std::min<size_t>(static_cast<size_t>(total), targetSampleCount));

    for (int y = 0; y < slice.rows; y += sampleStride)
    {
        const float* rowPtr = slice.ptr<float>(y);
        for (int x = 0; x < slice.cols; x += sampleStride)
        {
            const float value = rowPtr[x];
            if (std::isfinite(value))
            {
                outSamples.push_back(value);
            }
        }
    }
}

PreviewWindow estimateWindowFromSamples(std::vector<float>& samples)
{
    if (samples.empty())
    {
        return {};
    }

    const size_t lowerIndex = static_cast<size_t>((samples.size() - 1) * (kLowerPercentile / 100.0));
    const size_t upperIndex = static_cast<size_t>((samples.size() - 1) * (kUpperPercentile / 100.0));

    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(lowerIndex), samples.end());
    const double lower = samples[lowerIndex];

    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(upperIndex), samples.end());
    const double upper = samples[upperIndex];

    if (upper <= lower)
    {
        return {};
    }

    return { lower, upper };
}

} // namespace

PreviewWindow PreviewSliceRenderer::estimateRobustWindow(const cv::Mat& slice)
{
    if (slice.empty())
    {
        return {};
    }

    const int rows = slice.rows;
    const int cols = slice.cols;
    const int total = rows * cols;
    if (total <= 0)
    {
        return {};
    }

    std::vector<float> samples;
    appendFiniteSamples(slice, kMaxSamples, samples);

    if (samples.empty())
    {
        return minMaxWindow(slice);
    }

    const PreviewWindow estimated = estimateWindowFromSamples(samples);
    if (!estimated.isValid())
    {
        return minMaxWindow(slice);
    }

    return estimated;
}

PreviewWindow PreviewSliceRenderer::estimateRobustWindow(const std::vector<cv::Mat>& slices)
{
    std::vector<float> samples;
    if (slices.empty())
    {
        return {};
    }

    const size_t perSliceBudget = std::max<size_t>(1, kMaxSamples / std::max<size_t>(1, slices.size()));
    for (const cv::Mat& slice : slices)
    {
        appendFiniteSamples(slice, perSliceBudget, samples);
    }

    return estimateWindowFromSamples(samples);
}

QImage PreviewSliceRenderer::renderFloatSlice(const cv::Mat& slice)
{
    return renderFloatSlice(slice, estimateRobustWindow(slice));
}

QImage PreviewSliceRenderer::renderFloatSlice(const cv::Mat& slice, const PreviewWindow& window)
{
    if (slice.empty())
    {
        return {};
    }

    const PreviewWindow effectiveWindow = window.isValid() ? window : minMaxWindow(slice);
    const double range = effectiveWindow.upper - effectiveWindow.lower;

    cv::Mat normalized;
    slice.convertTo(normalized, CV_8U, 255.0 / range, -effectiveWindow.lower * 255.0 / range);

    return QImage(
        normalized.data,
        normalized.cols,
        normalized.rows,
        static_cast<int>(normalized.step),
        QImage::Format_Grayscale8).copy();
}

} // namespace ct
