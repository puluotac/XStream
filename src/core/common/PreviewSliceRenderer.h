#pragma once

#include <QImage>

#include <opencv2/core.hpp>

#include <vector>

namespace ct
{

struct PreviewWindow
{
    double lower = 0.0;
    double upper = 0.0;

    bool isValid() const
    {
        return upper > lower;
    }
};

class PreviewSliceRenderer
{
public:
    static PreviewWindow estimateRobustWindow(const cv::Mat& slice);
    static PreviewWindow estimateRobustWindow(const std::vector<cv::Mat>& slices);
    static QImage renderFloatSlice(const cv::Mat& slice);
    static QImage renderFloatSlice(const cv::Mat& slice, const PreviewWindow& window);
};

} // namespace ct
