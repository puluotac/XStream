#pragma once

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cstddef>
#include <cstdint>

namespace ct
{

void launchOfflineFbpBackprojection(const float* sinoSpatial, const float2* geom,
    int angleCount, int volW, int zChunkHeight, float corShift, float* outVolume,
    cudaStream_t stream);

void launchGridrecAccumulate(const cufftComplex* sinoFreq, const double* vectors,
    int angleCount, int zChunkHeight, int freqWidth, int gridSize, int paddedWidth,
    cufftComplex* gridVolume, cudaStream_t stream);

void launchGridrecIfftshiftCopy(const cufftComplex* src, int zChunkHeight,
    int gridH, int gridW, cufftComplex* dst, cudaStream_t stream);

void launchGridrecFinalizeCrop(const cufftComplex* imgComplex, const float* deapodMap,
    int zChunkHeight, int gridH, int gridW, int outH, int outW, int cropStart,
    float scalingFactor, float* outVolume, cudaStream_t stream);

void launchConvertFloatVolumeToUint16(const float* src, uint16_t* dst,
    size_t voxelCount, float vmin, float vmax, cudaStream_t stream);

} // namespace ct
