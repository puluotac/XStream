#pragma once

#include "core/common/ProjectionTypes.h"

namespace ct
{

class CorEstimatorFft
{
public:
    float estimateShift(const cv::Mat& img0, const cv::Mat& img180, const CorrectionContext& context) const;
    float estimateShift(const cv::Mat& img0, const CorrectionContext& context0,
        const cv::Mat& img180, const CorrectionContext& context180) const;

private:
    cv::Mat preprocessProjection(const cv::Mat& raw, const CorrectionContext& context, bool flip) const;
};

} // namespace ct
