#pragma once

#include "core/config/AppConfig.h"

#include <atomic>
#include <functional>
#include <memory>

#include <opencv2/core.hpp>

namespace ct
{

class IReconstructionVolume;

struct OfflineTomoResult
{
    std::shared_ptr<IReconstructionVolume> volume;
    AppConfig config;
};

class OfflineTomoReconstruction
{
public:
    using LogCallback = std::function<void(const QString&)>;
    using ProgressCallback = std::function<void(int, int)>;
    using CorCallback = std::function<void(double)>;
    using GeometryCallback = std::function<void(int, int)>;
    using PreviewCallback = std::function<void(bool)>;
    using DirectPreviewCallback = std::function<void(const cv::Mat&, bool)>;

    static OfflineTomoResult run(const AppConfig& inputConfig,
        std::atomic_bool& stopRequested,
        const LogCallback& log,
        const ProgressCallback& progress,
        const CorCallback& corComputed,
        const GeometryCallback& geometryResolved,
        const PreviewCallback& previewSuggested,
        const DirectPreviewCallback& directPreview);
};

} // namespace ct
