#pragma once

#include "core/config/AppConfig.h"
#include "core/recon/GpuProjectionBatch.h"

#include <opencv2/core.hpp>

#include <cuda_runtime_api.h>

namespace ct
{

struct DeviceCorrectionImages
{
    DeviceCorrectionImages() = default;
    ~DeviceCorrectionImages();
    DeviceCorrectionImages(const DeviceCorrectionImages&) = delete;
    DeviceCorrectionImages& operator=(const DeviceCorrectionImages&) = delete;
    DeviceCorrectionImages(DeviceCorrectionImages&& other) noexcept;
    DeviceCorrectionImages& operator=(DeviceCorrectionImages&& other) noexcept;

    float* flat = nullptr;
    float* dark = nullptr;
    int height = 0;
    int width = 0;

    void release();
};

class ImageNormalizer
{
public:
    static bool uploadCorrectionImages(const cv::Mat& flat, const cv::Mat& dark,
        int height, int width, DeviceCorrectionImages& outImages);
    static bool applyCorrection(GpuProjectionBatch& batch, const DeviceCorrectionImages& correctionImages,
        CorrectionMode mode, cudaStream_t stream);
    static bool applyCorrectionRange(GpuProjectionBatch& batch, int angleOffset, int angleCount,
        const DeviceCorrectionImages& correctionImages, CorrectionMode mode, cudaStream_t stream);
    static bool transposeRawToRecon(GpuProjectionBatch& batch, cudaStream_t stream);
    static bool applyMethodAndTaper(GpuProjectionBatch& batch, const PreprocessConfig& config, cudaStream_t stream);
};

} // namespace ct
