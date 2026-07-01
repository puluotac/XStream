#pragma once

#include <QString>

#include <opencv2/core.hpp>

namespace ct
{

cv::Mat readImageFile(const QString& filePath);
bool writeImageFile(const QString& filePath, const cv::Mat& image, QString* errorMessage = nullptr);

} // namespace ct
