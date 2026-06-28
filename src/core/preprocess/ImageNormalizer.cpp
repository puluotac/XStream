#include "core/preprocess/ImageNormalizer.h"

#include "cuda/preprocess_kernels.h"

#include <cuda_runtime.h>

namespace ct
{

namespace
{

bool uploadImageToDevice(const cv::Mat& input, int height, int width, float*& deviceImage)
{
    if (input.empty())
    {
        deviceImage = nullptr;
        return true;
    }

    if (input.rows != height || input.cols != width || input.channels() != 1)
    {
        return false;
    }

    cv::Mat floatImage;
    input.convertTo(floatImage, CV_32F);

    if (cudaMalloc(&deviceImage, static_cast<size_t>(height) * width * sizeof(float)) != cudaSuccess)
    {
        deviceImage = nullptr;
        return false;
    }

    if (cudaMemcpy(deviceImage, floatImage.ptr<float>(), static_cast<size_t>(height) * width * sizeof(float),
        cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(deviceImage);
        deviceImage = nullptr;
        return false;
    }

    return true;
}

} // namespace

DeviceCorrectionImages::~DeviceCorrectionImages()
{
    release();
}

DeviceCorrectionImages::DeviceCorrectionImages(DeviceCorrectionImages&& other) noexcept
    : flat(other.flat)
    , dark(other.dark)
    , height(other.height)
    , width(other.width)
{
    other.flat = nullptr;
    other.dark = nullptr;
    other.height = 0;
    other.width = 0;
}

DeviceCorrectionImages& DeviceCorrectionImages::operator=(DeviceCorrectionImages&& other) noexcept
{
    if (this != &other)
    {
        release();
        flat = other.flat;
        dark = other.dark;
        height = other.height;
        width = other.width;
        other.flat = nullptr;
        other.dark = nullptr;
        other.height = 0;
        other.width = 0;
    }
    return *this;
}

void DeviceCorrectionImages::release()
{
    if (flat != nullptr)
    {
        cudaFree(flat);
        flat = nullptr;
    }
    if (dark != nullptr)
    {
        cudaFree(dark);
        dark = nullptr;
    }
    height = 0;
    width = 0;
}

bool ImageNormalizer::uploadCorrectionImages(const cv::Mat& flat, const cv::Mat& dark,
    int height, int width, DeviceCorrectionImages& outImages)
{
    outImages.release();
    outImages.height = height;
    outImages.width = width;

    if (!uploadImageToDevice(flat, height, width, outImages.flat))
    {
        outImages.release();
        return false;
    }

    if (!uploadImageToDevice(dark, height, width, outImages.dark))
    {
        outImages.release();
        return false;
    }

    return true;
}

bool ImageNormalizer::applyCorrection(GpuProjectionBatch& batch, const DeviceCorrectionImages& correctionImages,
    CorrectionMode mode, cudaStream_t stream)
{
    if (batch.rawAngleMajor() == nullptr || batch.currentAngleCount() <= 0)
    {
        return false;
    }

    return launchApplyCorrection(batch.rawAngleMajor(), batch.currentAngleCount(),
        batch.height(), batch.width(), correctionImages.flat, correctionImages.dark,
        static_cast<int>(mode), stream);
}

bool ImageNormalizer::applyCorrectionRange(GpuProjectionBatch& batch, int angleOffset, int angleCount,
    const DeviceCorrectionImages& correctionImages, CorrectionMode mode, cudaStream_t stream)
{
    if (batch.rawAngleMajor() == nullptr || batch.currentAngleCount() <= 0
        || angleOffset < 0 || angleCount <= 0 || angleOffset + angleCount > batch.currentAngleCount())
    {
        return false;
    }

    return launchApplyCorrectionRange(batch.rawAngleMajor(), angleOffset, angleCount,
        batch.height(), batch.width(), correctionImages.flat, correctionImages.dark,
        static_cast<int>(mode), stream);
}

bool ImageNormalizer::transposeRawToRecon(GpuProjectionBatch& batch, cudaStream_t stream)
{
    if (batch.rawAngleMajor() == nullptr || batch.reconLayout() == nullptr || batch.currentAngleCount() <= 0)
    {
        return false;
    }

    return launchTransposeAngleMajorToRecon(batch.rawAngleMajor(), batch.reconLayout(),
        batch.currentAngleCount(), batch.height(), batch.width(), stream);
}

bool ImageNormalizer::applyMethodAndTaper(GpuProjectionBatch& batch, const PreprocessConfig& config, cudaStream_t stream)
{
    if (batch.reconLayout() == nullptr || batch.currentAngleCount() <= 0)
    {
        return false;
    }

    return launchApplyMethodAndTaper(batch.reconLayout(), batch.height(), batch.currentAngleCount(),
        batch.width(), static_cast<int>(config.method), config.airValue, config.taperWidth, stream);
}

} // namespace ct
