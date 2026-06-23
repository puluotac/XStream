#include "core/common/ProjectionTypes.h"
#include "core/common/GeometryVectorGenerator.h"
#include "core/config/AppConfig.h"
#include "core/cor/CorEstimatorFft.h"
#include "core/preprocess/ImageNormalizer.h"
#include "core/recon/GpuProjectionBatch.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    const ct::AppConfig defaults = ct::AppConfig::defaults();
    const ct::AppConfig roundTrip = ct::AppConfig::fromJsonObject(defaults.toJsonObject());
    if (roundTrip.compute.batchSize != defaults.compute.batchSize
        || roundTrip.compute.zChunkSize != defaults.compute.zChunkSize
        || roundTrip.compute.gpuIndex != defaults.compute.gpuIndex)
    {
        std::cerr << "Compute config round-trip mismatch\n";
        return 3;
    }
    if (defaults.toJsonObject().value("COMPUTE").toObject().contains("BP_KERNEL_MODE"))
    {
        std::cerr << "Unexpected legacy BP kernel field in new config JSON\n";
        return 11;
    }

    QTemporaryDir tempDir;
    if (!tempDir.isValid())
    {
        std::cerr << "Failed to create temp directory\n";
        return 4;
    }

    ct::AppConfig sessionConfig = defaults;
    sessionConfig.geometry.corShift = 12.5f;
    sessionConfig.phase.enable = true;
    sessionConfig.phase.energyKev = 18.0f;
    sessionConfig.compute.batchSize = 32;
    sessionConfig.preview.enableLive = false;
    sessionConfig.io.watchDir = "E:/demo/watch";
    sessionConfig.io.saveDir = "E:/demo/save";
    sessionConfig.motorAxes["R"].speed = 2.5;
    sessionConfig.motorAxes["R"].accel = 10.0;
    sessionConfig.motorAxes["Y"].speed = 500.0;
    sessionConfig.motorAxes["Y"].accel = 2000.0;

    QJsonObject legacyRoot = sessionConfig.toJsonObject();
    QJsonObject legacyCompute = legacyRoot.value("COMPUTE").toObject();
    legacyCompute["BP_KERNEL_MODE"] = "baseline";
    legacyRoot["COMPUTE"] = legacyCompute;

    const QString sessionPath = tempDir.filePath("legacy_session.json");
    QFile sessionFile(sessionPath);
    if (!sessionFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        std::cerr << "Failed to create legacy session config\n";
        return 5;
    }
    sessionFile.write(QJsonDocument(legacyRoot).toJson(QJsonDocument::Indented));
    sessionFile.close();

    ct::AppConfig loadedConfig = ct::AppConfig::defaults();
    QString loadError;
    if (!ct::AppConfig::tryLoadFromJsonFile(sessionPath, &loadedConfig, &loadError))
    {
        std::cerr << "Failed to load saved session config: " << loadError.toStdString() << "\n";
        return 6;
    }

    if (!loadedConfig.phase.enable
        || loadedConfig.compute.batchSize != sessionConfig.compute.batchSize
        || loadedConfig.io.watchDir != sessionConfig.io.watchDir
        || std::abs(loadedConfig.geometry.corShift - sessionConfig.geometry.corShift) > 1e-6f
        || loadedConfig.preview.enableLive != sessionConfig.preview.enableLive
        || std::abs(loadedConfig.motorAxes.value("R").speed - 2.5) > 1e-9
        || std::abs(loadedConfig.motorAxes.value("R").accel - 10.0) > 1e-9
        || std::abs(loadedConfig.motorAxes.value("Y").speed - 500.0) > 1e-9
        || std::abs(loadedConfig.motorAxes.value("Y").accel - 2000.0) > 1e-9)
    {
        std::cerr << "Session config round-trip mismatch\n";
        return 7;
    }

    const QString cleanedPath = tempDir.filePath("cleaned_session.json");
    if (!loadedConfig.saveToJsonFile(cleanedPath))
    {
        std::cerr << "Failed to save cleaned session config\n";
        return 12;
    }

    QFile cleanedFile(cleanedPath);
    if (!cleanedFile.open(QIODevice::ReadOnly))
    {
        std::cerr << "Failed to re-open cleaned session config\n";
        return 13;
    }
    const QJsonDocument cleanedDoc = QJsonDocument::fromJson(cleanedFile.readAll());
    cleanedFile.close();
    if (cleanedDoc.object().value("COMPUTE").toObject().contains("BP_KERNEL_MODE"))
    {
        std::cerr << "Legacy BP kernel field was not stripped on save\n";
        return 14;
    }
    if (!cleanedDoc.object().value("MOTOR").toObject().contains("R")
        || !cleanedDoc.object().value("MOTOR").toObject().contains("Y"))
    {
        std::cerr << "Motor speed/accel config was not saved\n";
        return 21;
    }

    const QString invalidPath = tempDir.filePath("invalid.json");
    QFile invalidFile(invalidPath);
    if (!invalidFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        std::cerr << "Failed to create invalid JSON file\n";
        return 8;
    }
    invalidFile.write("{ invalid json");
    invalidFile.close();

    QString invalidError;
    if (ct::AppConfig::tryLoadFromJsonFile(invalidPath, &loadedConfig, &invalidError))
    {
        std::cerr << "Invalid JSON unexpectedly loaded\n";
        return 9;
    }
    if (invalidError.isEmpty())
    {
        std::cerr << "Invalid JSON did not return an error message\n";
        return 10;
    }

    const auto vectors = ct::GeometryVectorGenerator::createParallel3dVectors(defaults.geometry);
    if (vectors.size() != static_cast<size_t>(defaults.geometry.totalAngles) * 12)
    {
        std::cerr << "Geometry vector size mismatch\n";
        return 1;
    }

    constexpr int width = 192;
    constexpr int height = 160;
    constexpr float detectorShiftPixels = -20.0f;

    cv::Mat attenuation(height, width, CV_32F, cv::Scalar(0.2f));
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const float gx = static_cast<float>(x - width / 2) / 30.0f;
            const float gy = static_cast<float>(y - height / 2) / 25.0f;
            attenuation.at<float>(y, x) = 0.4f + 0.9f * std::exp(-(gx * gx + gy * gy)) + 0.15f * std::sin(x * 0.13f);
        }
    }

    cv::Mat shifted;
    cv::Mat affine = (cv::Mat_<double>(2, 3) << 1.0, 0.0, detectorShiftPixels, 0.0, 1.0, 0.0);
    cv::warpAffine(attenuation, shifted, affine, attenuation.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Mat raw0;
    cv::Mat raw180PreFlip;
    cv::exp(-attenuation, raw0);
    cv::exp(-shifted, raw180PreFlip);
    raw0 *= 65535.0f;
    raw180PreFlip *= 65535.0f;

    ct::CorrectionContext context;
    context.mode = ct::CorrectionMode::None;
    context.cropMargin = 0;

    ct::CorEstimatorFft estimator;
    const float estimatedShift = estimator.estimateShift(raw0, raw180PreFlip, context);
    const float expectedCor = detectorShiftPixels * 0.5f;
    if (std::abs(estimatedShift - expectedCor) > 0.5f)
    {
        std::cerr << "COR estimator mismatch: expected " << expectedCor
                  << " got " << estimatedShift << "\n";
        return 2;
    }

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0)
    {
        std::cerr << "No CUDA device available for preprocessing test\n";
        return 15;
    }
    if (cudaSetDevice(defaults.compute.gpuIndex) != cudaSuccess)
    {
        std::cerr << "Failed to select CUDA device for preprocessing test\n";
        return 16;
    }

    ct::HostPinnedProjectionBatch hostBatch;
    ct::GpuProjectionBatch gpuBatch;
    constexpr int testHeight = 2;
    constexpr int testWidth = 3;
    if (!hostBatch.initialize(1, testHeight, testWidth) || !gpuBatch.initialize(1, testHeight, testWidth))
    {
        std::cerr << "Failed to initialize GPU preprocessing buffers\n";
        return 17;
    }

    cv::Mat raw(testHeight, testWidth, CV_32F);
    cv::Mat flatRef(testHeight, testWidth, CV_32F, cv::Scalar(20.0f));
    cv::Mat darkRef(testHeight, testWidth, CV_32F, cv::Scalar(2.0f));
    for (int y = 0; y < testHeight; ++y)
    {
        for (int x = 0; x < testWidth; ++x)
        {
            raw.at<float>(y, x) = static_cast<float>(10 + y * testWidth + x);
        }
    }

    ct::DeviceCorrectionImages correctionImages;
    if (!hostBatch.copyFrame(0, raw)
        || !gpuBatch.uploadRawFromHostAsync(hostBatch, 1, nullptr)
        || !ct::ImageNormalizer::uploadCorrectionImages(flatRef, darkRef, testHeight, testWidth, correctionImages)
        || !ct::ImageNormalizer::applyCorrection(gpuBatch, correctionImages, ct::CorrectionMode::FlatDark, nullptr)
        || cudaDeviceSynchronize() != cudaSuccess)
    {
        const cudaError_t status = cudaGetLastError();
        correctionImages.release();
        std::cerr << "GPU flat/dark correction smoke test failed: " << cudaGetErrorString(status) << "\n";
        return 18;
    }

    std::vector<float> corrected(static_cast<size_t>(testHeight) * testWidth);
    if (cudaMemcpy(corrected.data(), gpuBatch.rawAngleMajor(),
        corrected.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        correctionImages.release();
        std::cerr << "Failed to download GPU correction smoke test result\n";
        return 19;
    }
    correctionImages.release();

    for (size_t index = 0; index < corrected.size(); ++index)
    {
        const float expected = (raw.ptr<float>()[index] - 2.0f) / 18.0f;
        if (std::abs(corrected[index] - expected) > 1e-5f)
        {
            std::cerr << "GPU correction mismatch at " << index
                      << ": expected " << expected
                      << " got " << corrected[index] << "\n";
            return 20;
        }
    }

    std::cout << "ct_workstation_tests passed\n";
    return 0;
}
