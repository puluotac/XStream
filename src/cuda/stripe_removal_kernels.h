#pragma once

#include <cuda_runtime_api.h>
#include <cufft.h>

namespace ct
{

bool launchPackSinogramColumnsForStripe(const float* sino, float* packed,
    int zCount, int angleCount, int width, cudaStream_t stream);

bool launchPadStripeInput(const float* sino, float* padded,
    int zCount, int angleCount, int paddedAngleCount, int width, int angleShift, cudaStream_t stream);

bool launchDampStripeSpectrum(cufftComplex* spectrum,
    int vectorCount, int spectrumLength, int angleCount, float sigma, cudaStream_t stream);

bool launchUnpackSinogramColumnsForStripe(const float* packed, float* sino,
    int zCount, int angleCount, int width, cudaStream_t stream);

bool launchCropStripeOutput(const float* padded, float* sino,
    int zCount, int angleCount, int paddedAngleCount, int width, int angleShift, cudaStream_t stream);

bool launchDwtAxis1Db5(const float* src, float* approx, float* detail,
    int zCount, int dataLen, int width, int outLen, int paddedLen, cudaStream_t stream);

bool launchDwtAxis2Db5(const float* src, float* approx, float* detail,
    int zCount, int height, int dataLen, int outLen, int paddedLen, cudaStream_t stream);

bool launchIdwtAxis1Db5(const float* approx, const float* detail, float* out,
    int zCount, int coeffLen, int width, int targetLen, cudaStream_t stream);

bool launchIdwtAxis2Db5(const float* approx, const float* detail, float* out,
    int zCount, int height, int coeffLen, int targetLen, cudaStream_t stream);

} // namespace ct
