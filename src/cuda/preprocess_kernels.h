#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>
#include <cufft.h>

namespace ct
{

bool launchApplyCorrection(float* rawAngleMajor, int angleCount, int height, int width,
    const float* flat, const float* dark, int correctionMode, cudaStream_t stream);

bool launchTransposeAngleMajorToRecon(const float* rawAngleMajor, float* reconLayout,
    int angleCount, int height, int width, cudaStream_t stream);

bool launchApplyMethodAndTaper(float* reconLayout, int height, int angleCount, int width,
    int preprocessMethod, float airValue, int taperWidth, cudaStream_t stream);

bool launchPadPaganinInputReflectY(const float* rawAngleMajor, float* paddedInput,
    int angleCount, int height, int paddedHeight, int width, int padTop, cudaStream_t stream);

bool launchMultiplyPaganinSpectrum(cufftComplex* spectrum, const float* filterKernel,
    int batchAngles, int paddedHeight, int spectrumWidth, cudaStream_t stream);

bool launchNormalizeClamp(float* data, size_t elementCount, float scale, float clampMin, cudaStream_t stream);

bool launchCropPaganinOutputY(const float* paddedInput, float* rawAngleMajor,
    int angleCount, int height, int paddedHeight, int width, int padTop, float scale,
    float clampMin, cudaStream_t stream);

bool launchPrepareFbpInput(const float* reconLayout, float* paddedInput,
    int transformCount, int width, int paddedLength, cudaStream_t stream);

bool launchMultiplyFbpSpectrum(cufftComplex* spectrum, const float* filterSpectrum,
    int transformCount, int spectrumLength, cudaStream_t stream);

bool launchExtractFbpOutput(const float* paddedOutput, float* reconLayout,
    int transformCount, int width, int paddedLength, float scale, cudaStream_t stream);

} // namespace ct
