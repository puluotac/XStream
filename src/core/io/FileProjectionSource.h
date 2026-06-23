#pragma once

#include "core/common/ProjectionTypes.h"
#include "core/config/AppConfig.h"

#include <atomic>
#include <vector>

namespace ct
{

class FileProjectionSource
{
public:
    explicit FileProjectionSource(IoConfig config);

    int extractStartIndex(const QString& pattern) const;
    QString makeNumberedFileName(const QString& prefix, int index) const;

    bool waitForReferenceImage(const QString& prefix, const QString& label, cv::Mat& outImage,
        std::atomic_bool& stopRequested) const;
    bool waitForSpecificFile(const QString& filePattern, const QString& label, cv::Mat& outImage,
        QString* outPath, std::atomic_bool& stopRequested) const;

    std::vector<ProjectionFrame> pollNextBatch(const std::vector<bool>& processedFlags,
        int startIndex, int maxBatchSize, std::atomic_bool& stopRequested) const;

    cv::Mat readImageRobust(const QString& filePath) const;

private:
    IoConfig config_;
};

} // namespace ct
