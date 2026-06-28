#include "core/cor/CorEstimatorFft.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <vector>

namespace ct
{

cv::Mat CorEstimatorFft::preprocessProjection(
    const cv::Mat& raw,
    const CorrectionContext& context,
    bool flip) const
{
    cv::Mat rawFloat;
    raw.convertTo(rawFloat, CV_32F);

    cv::Mat normalized = rawFloat.clone();

    if (context.mode != CorrectionMode::None)
    {
        cv::Mat flat = context.flat.empty() ? cv::Mat(raw.size(), CV_32F, cv::Scalar(65535.0f)) : context.flat;
        cv::Mat dark = context.dark.empty() ? cv::Mat(raw.size(), CV_32F, cv::Scalar(0.0f)) : context.dark;

        if (flat.type() != CV_32F)
        {
            flat.convertTo(flat, CV_32F);
        }
        if (dark.type() != CV_32F)
        {
            dark.convertTo(dark, CV_32F);
        }

        cv::Mat numerator = rawFloat - dark;
        cv::Mat denominator = flat - dark + 1e-6f;
        cv::divide(numerator, denominator, normalized);
    }
    else
    {
        normalized /= 65535.0f;
    }

    cv::max(normalized, 1e-6f, normalized);
    cv::log(normalized, normalized);
    normalized *= -1.0f;

    if (flip)
    {
        cv::flip(normalized, normalized, 1);
    }

    const int margin = context.cropMargin;
    if (normalized.rows > 2 * margin && normalized.cols > 2 * margin)
    {
        normalized = normalized(cv::Rect(
            margin,
            margin,
            normalized.cols - 2 * margin,
            normalized.rows - 2 * margin)).clone();
    }

    return normalized;
}

float CorEstimatorFft::estimateShift(const cv::Mat& img0, const cv::Mat& img180, const CorrectionContext& context) const
{
    return estimateShift(img0, context, img180, context);
}

float CorEstimatorFft::estimateShift(const cv::Mat& img0, const CorrectionContext& context0,
    const cv::Mat& img180, const CorrectionContext& context180) const
{
    cv::Mat pre0 = preprocessProjection(img0, context0, false);
    cv::Mat pre180 = preprocessProjection(img180, context180, true);

    cv::Mat f0;
    cv::Mat f1;
    cv::dft(pre0, f0, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(pre180, f1, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat crossPower;
    cv::mulSpectrums(f0, f1, crossPower, 0, true);

    std::vector<cv::Mat> crossChannels;
    cv::split(crossPower, crossChannels);
    cv::Mat magnitude;
    cv::magnitude(crossChannels[0], crossChannels[1], magnitude);
    magnitude += 1e-9f;
    crossChannels[0] /= magnitude;
    crossChannels[1] /= magnitude;
    cv::merge(crossChannels, crossPower);

    cv::Mat correlationComplex;
    cv::dft(crossPower, correlationComplex, cv::DFT_INVERSE | cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);

    std::vector<cv::Mat> corrChannels;
    cv::split(correlationComplex, corrChannels);
    cv::Mat correlationAbs;
    cv::magnitude(corrChannels[0], corrChannels[1], correlationAbs);

    double peakValue = 0.0;
    cv::Point peakLoc;
    cv::minMaxLoc(correlationAbs, nullptr, &peakValue, nullptr, &peakLoc);

    double peakX = static_cast<double>(peakLoc.x);
    const int width = correlationAbs.cols;

    if (peakLoc.x > 0 && peakLoc.x < (width - 1))
    {
        const float left = correlationAbs.at<float>(peakLoc.y, peakLoc.x - 1);
        const float center = correlationAbs.at<float>(peakLoc.y, peakLoc.x);
        const float right = correlationAbs.at<float>(peakLoc.y, peakLoc.x + 1);
        const double denom = 2.0 * (static_cast<double>(left) - 2.0 * static_cast<double>(center) + static_cast<double>(right));
        if (std::abs(denom) > 1e-9)
        {
            const double delta = (static_cast<double>(left) - static_cast<double>(right)) / denom;
            if (delta >= -1.0 && delta <= 1.0)
            {
                peakX += delta;
            }
        }
    }

    double shiftX = peakX;
    if (shiftX > static_cast<double>(width) / 2.0)
    {
        shiftX -= static_cast<double>(width);
    }

    return static_cast<float>(shiftX * 0.5);
}

} // namespace ct
