#include "core/common/ProjectionTypes.h"
#include "core/common/GeometryVectorGenerator.h"
#include "core/config/AppConfig.h"
#include "core/cor/CorEstimatorFft.h"
#include "core/preprocess/ImageNormalizer.h"
#include "core/recon/GpuProjectionBatch.h"
#include "cuda/stripe_removal_kernels.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cufft.h>
#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    const ct::AppConfig defaults = ct::AppConfig::defaults();
    const ct::AppConfig roundTrip = ct::AppConfig::fromJsonObject(defaults.toJsonObject());
    if (roundTrip.compute.batchSize != defaults.compute.batchSize
        || roundTrip.compute.zChunkSize != defaults.compute.zChunkSize
        || roundTrip.compute.gpuIndex != defaults.compute.gpuIndex
        || roundTrip.stripe.method != defaults.stripe.method
        || roundTrip.stripe.enable != defaults.stripe.enable)
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
    sessionConfig.stripe.enable = true;
    sessionConfig.stripe.level = 1;
    sessionConfig.stripe.sigma = 2.5f;
    sessionConfig.compute.batchSize = 32;
    sessionConfig.preview.enableLive = false;
    sessionConfig.cameraUi.exposureValue = 123.0;
    sessionConfig.cameraUi.exposureUnit = "ms";
    sessionConfig.cameraUi.roiX = 11;
    sessionConfig.cameraUi.roiY = 12;
    sessionConfig.cameraUi.roiWidth = 1024;
    sessionConfig.cameraUi.roiHeight = 768;
    sessionConfig.cameraUi.snapPrefix = "snap_";
    sessionConfig.cameraUi.snapPadding = 5;
    sessionConfig.cameraUi.snapIndex = 42;
    sessionConfig.cameraUi.saveDir = "E:/demo/camera";
    sessionConfig.ctAcquisitionUi.beginningAngleDeg = -12.5;
    sessionConfig.ctAcquisitionUi.modeIndex = 1;
    sessionConfig.ctAcquisitionUi.savePath = "E:/demo/ct";
    sessionConfig.ctAcquisitionUi.projectionNum = 1440;
    sessionConfig.ctAcquisitionUi.flatGap = 180;
    sessionConfig.ctAcquisitionUi.flatMoveUm = 10000.0;
    sessionConfig.ctAcquisitionUi.flatNum = 2;
    sessionConfig.ctAcquisitionUi.flyLeadAngleDeg = 3.0;
    sessionConfig.ctAcquisitionUi.tiffThreads = 24;
    sessionConfig.ctFly.allowedBlurPx = 1.25;
    sessionConfig.kohzuUi.host = "10.30.92.202";
    sessionConfig.kohzuUi.port = 12322;
    sessionConfig.io.watchDir = "E:/demo/watch";
    sessionConfig.io.saveDir = "E:/demo/save";
    sessionConfig.motorAxes["R"].hasBaseSpeed = true;
    sessionConfig.motorAxes["R"].hasRelativeStep = true;
    sessionConfig.motorAxes["R"].relativeStep = 0.25;
    sessionConfig.motorAxes["R"].hasAbsoluteTarget = true;
    sessionConfig.motorAxes["R"].absoluteTarget = 90.0;
    sessionConfig.motorAxes["R"].baseSpeed = 1.0;
    sessionConfig.motorAxes["R"].speed = 2.5;
    sessionConfig.motorAxes["R"].accel = 10.0;
    sessionConfig.motorAxes["Y"].hasBaseSpeed = true;
    sessionConfig.motorAxes["Y"].hasRelativeStep = true;
    sessionConfig.motorAxes["Y"].relativeStep = 500.0;
    sessionConfig.motorAxes["Y"].hasAbsoluteTarget = true;
    sessionConfig.motorAxes["Y"].absoluteTarget = -1000.0;
    sessionConfig.motorAxes["Y"].baseSpeed = 100.0;
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
        || loadedConfig.stripe.enable
        || loadedConfig.stripe.level != sessionConfig.stripe.level
        || std::abs(loadedConfig.stripe.sigma - sessionConfig.stripe.sigma) > 1e-6f
        || loadedConfig.io.watchDir != sessionConfig.io.watchDir
        || std::abs(loadedConfig.geometry.corShift - sessionConfig.geometry.corShift) > 1e-6f
        || loadedConfig.preview.enableLive != sessionConfig.preview.enableLive
        || std::abs(loadedConfig.cameraUi.exposureValue - sessionConfig.cameraUi.exposureValue) > 1e-9
        || loadedConfig.cameraUi.exposureUnit != sessionConfig.cameraUi.exposureUnit
        || loadedConfig.cameraUi.roiX != sessionConfig.cameraUi.roiX
        || loadedConfig.cameraUi.roiY != sessionConfig.cameraUi.roiY
        || loadedConfig.cameraUi.roiWidth != sessionConfig.cameraUi.roiWidth
        || loadedConfig.cameraUi.roiHeight != sessionConfig.cameraUi.roiHeight
        || loadedConfig.cameraUi.snapPrefix != sessionConfig.cameraUi.snapPrefix
        || loadedConfig.cameraUi.snapPadding != sessionConfig.cameraUi.snapPadding
        || loadedConfig.cameraUi.snapIndex != sessionConfig.cameraUi.snapIndex
        || loadedConfig.cameraUi.saveDir != sessionConfig.cameraUi.saveDir
        || std::abs(loadedConfig.ctAcquisitionUi.beginningAngleDeg - sessionConfig.ctAcquisitionUi.beginningAngleDeg) > 1e-9
        || loadedConfig.ctAcquisitionUi.modeIndex != sessionConfig.ctAcquisitionUi.modeIndex
        || loadedConfig.ctAcquisitionUi.savePath != sessionConfig.ctAcquisitionUi.savePath
        || loadedConfig.ctAcquisitionUi.projectionNum != sessionConfig.ctAcquisitionUi.projectionNum
        || loadedConfig.ctAcquisitionUi.flatGap != sessionConfig.ctAcquisitionUi.flatGap
        || std::abs(loadedConfig.ctAcquisitionUi.flatMoveUm - sessionConfig.ctAcquisitionUi.flatMoveUm) > 1e-9
        || loadedConfig.ctAcquisitionUi.flatNum != sessionConfig.ctAcquisitionUi.flatNum
        || std::abs(loadedConfig.ctAcquisitionUi.flyLeadAngleDeg - sessionConfig.ctAcquisitionUi.flyLeadAngleDeg) > 1e-9
        || loadedConfig.ctAcquisitionUi.tiffThreads != sessionConfig.ctAcquisitionUi.tiffThreads
        || std::abs(loadedConfig.ctFly.allowedBlurPx - sessionConfig.ctFly.allowedBlurPx) > 1e-9
        || loadedConfig.kohzuUi.host != sessionConfig.kohzuUi.host
        || loadedConfig.kohzuUi.port != sessionConfig.kohzuUi.port
        || !loadedConfig.motorAxes.value("R").hasBaseSpeed
        || !loadedConfig.motorAxes.value("R").hasRelativeStep
        || std::abs(loadedConfig.motorAxes.value("R").relativeStep - 0.25) > 1e-9
        || !loadedConfig.motorAxes.value("R").hasAbsoluteTarget
        || std::abs(loadedConfig.motorAxes.value("R").absoluteTarget - 90.0) > 1e-9
        || std::abs(loadedConfig.motorAxes.value("R").baseSpeed - 1.0) > 1e-9
        || std::abs(loadedConfig.motorAxes.value("R").speed - 2.5) > 1e-9
        || std::abs(loadedConfig.motorAxes.value("R").accel - 10.0) > 1e-9
        || !loadedConfig.motorAxes.value("Y").hasBaseSpeed
        || !loadedConfig.motorAxes.value("Y").hasRelativeStep
        || std::abs(loadedConfig.motorAxes.value("Y").relativeStep - 500.0) > 1e-9
        || !loadedConfig.motorAxes.value("Y").hasAbsoluteTarget
        || std::abs(loadedConfig.motorAxes.value("Y").absoluteTarget + 1000.0) > 1e-9
        || std::abs(loadedConfig.motorAxes.value("Y").baseSpeed - 100.0) > 1e-9
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
    if (!cleanedDoc.object().value("MOTOR").toObject().value("R").toObject().contains("BASE_SPEED"))
    {
        std::cerr << "Motor base speed config was not saved\n";
        return 23;
    }
    if (!cleanedDoc.object().value("MOTOR").toObject().value("R").toObject().contains("RELATIVE_STEP")
        || !cleanedDoc.object().value("MOTOR").toObject().value("R").toObject().contains("ABSOLUTE_TARGET"))
    {
        std::cerr << "Motor motion input config was not saved\n";
        return 29;
    }
    QJsonObject oldMotorRoot;
    QJsonObject oldMotorAxis;
    oldMotorAxis["SPEED"] = 12.0;
    oldMotorAxis["ACCEL"] = 3.0;
    oldMotorRoot["R"] = oldMotorAxis;
    QJsonObject oldRoot;
    oldRoot["MOTOR"] = oldMotorRoot;
    const ct::AppConfig oldMotorConfig = ct::AppConfig::fromJsonObject(oldRoot);
    if (oldMotorConfig.motorAxes.value("R").hasBaseSpeed
        || std::abs(oldMotorConfig.motorAxes.value("R").speed - 12.0) > 1e-9
        || std::abs(oldMotorConfig.motorAxes.value("R").accel - 3.0) > 1e-9)
    {
        std::cerr << "Legacy motor config base speed compatibility failed\n";
        return 24;
    }
    if (cleanedDoc.object().value("STRIPE").toObject().value("ENABLE").toBool(false))
    {
        std::cerr << "Stripe config should be saved disabled during commissioning\n";
        return 22;
    }
    if (!cleanedDoc.object().contains("CAMERA")
        || !cleanedDoc.object().contains("CT_ACQUISITION")
        || !cleanedDoc.object().contains("KOHZU"))
    {
        std::cerr << "Left-side UI config objects were not saved\n";
        return 28;
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

    constexpr int stripeZ = 2;
    constexpr int stripeAngles = 8;
    constexpr int stripeWidth = 8;
    constexpr int paddedAngles = 9;
    constexpr int angleShift = 0;
    constexpr int angleCoeff = (paddedAngles + 9) / 2;
    constexpr int anglePadded = angleCoeff * 2;
    constexpr int widthCoeff = (stripeWidth + 9) / 2;
    constexpr int widthPadded = widthCoeff * 2;
    const size_t stripeInputCount = static_cast<size_t>(stripeZ) * stripeAngles * stripeWidth;
    const size_t paddedCount = static_cast<size_t>(stripeZ) * paddedAngles * stripeWidth;
    const size_t bandCount = static_cast<size_t>(stripeZ) * angleCoeff * widthCoeff;
    std::vector<float> stripeHost(stripeInputCount);
    for (int z = 0; z < stripeZ; ++z)
    {
        for (int a = 0; a < stripeAngles; ++a)
        {
            for (int x = 0; x < stripeWidth; ++x)
            {
                const float stripe = (x % 4 < 2) ? 0.25f : -0.25f;
                stripeHost[(static_cast<size_t>(z) * stripeAngles + a) * stripeWidth + x] =
                    1.0f + 0.02f * static_cast<float>(z) + 0.01f * static_cast<float>(a) + stripe;
            }
        }
    }
    float* stripeDevice = nullptr;
    float* paddedDevice = nullptr;
    float* tempA = nullptr;
    float* tempD = nullptr;
    float* approx = nullptr;
    float* cV = nullptr;
    float* cH = nullptr;
    float* cD = nullptr;
    float* packed = nullptr;
    cufftComplex* spectrum = nullptr;
    cufftHandle r2c = 0;
    cufftHandle c2r = 0;
    const int spectrumLength = angleCoeff / 2 + 1;
    const int vectorCount = stripeZ * widthCoeff;
    if (cudaMalloc(&stripeDevice, stripeInputCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&paddedDevice, paddedCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&tempA, paddedCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&tempD, paddedCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&approx, bandCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&cV, bandCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&cH, bandCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&cD, bandCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&packed, bandCount * sizeof(float)) != cudaSuccess
        || cudaMalloc(&spectrum, static_cast<size_t>(vectorCount) * spectrumLength * sizeof(cufftComplex)) != cudaSuccess)
    {
        std::cerr << "Failed to allocate stripe kernel smoke buffers\n";
        return 23;
    }
    if (cudaMemcpy(stripeDevice, stripeHost.data(), stripeInputCount * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || !ct::launchPadStripeInput(stripeDevice, paddedDevice, stripeZ, stripeAngles, paddedAngles, stripeWidth, angleShift, nullptr)
        || !ct::launchDwtAxis1Db5(paddedDevice, tempA, tempD, stripeZ, paddedAngles, stripeWidth, angleCoeff, anglePadded, nullptr)
        || !ct::launchDwtAxis2Db5(tempA, approx, cV, stripeZ, angleCoeff, stripeWidth, widthCoeff, widthPadded, nullptr)
        || !ct::launchDwtAxis2Db5(tempD, cH, cD, stripeZ, angleCoeff, stripeWidth, widthCoeff, widthPadded, nullptr)
        || !ct::launchPackSinogramColumnsForStripe(cV, packed, stripeZ, angleCoeff, widthCoeff, nullptr))
    {
        std::cerr << "Stripe FW CUDA smoke launch failed before FFT\n";
        return 24;
    }
    int n[1] = { angleCoeff };
    int inembed[1] = { angleCoeff };
    int onembed[1] = { spectrumLength };
    if (cufftPlanMany(&r2c, 1, n, inembed, 1, angleCoeff, onembed, 1, spectrumLength, CUFFT_R2C, vectorCount) != CUFFT_SUCCESS
        || cufftPlanMany(&c2r, 1, n, onembed, 1, spectrumLength, inembed, 1, angleCoeff, CUFFT_C2R, vectorCount) != CUFFT_SUCCESS
        || cufftExecR2C(r2c, reinterpret_cast<cufftReal*>(packed), spectrum) != CUFFT_SUCCESS
        || !ct::launchDampStripeSpectrum(spectrum, vectorCount, spectrumLength, angleCoeff, 2.0f, nullptr)
        || cufftExecC2R(c2r, spectrum, reinterpret_cast<cufftReal*>(packed)) != CUFFT_SUCCESS
        || !ct::launchUnpackSinogramColumnsForStripe(packed, cV, stripeZ, angleCoeff, widthCoeff, nullptr)
        || !ct::launchIdwtAxis2Db5(cH, cD, tempD, stripeZ, angleCoeff, widthCoeff, stripeWidth, nullptr)
        || !ct::launchIdwtAxis2Db5(approx, cV, tempA, stripeZ, angleCoeff, widthCoeff, stripeWidth, nullptr)
        || !ct::launchIdwtAxis1Db5(tempA, tempD, paddedDevice, stripeZ, angleCoeff, stripeWidth, paddedAngles, nullptr)
        || !ct::launchCropStripeOutput(paddedDevice, stripeDevice, stripeZ, stripeAngles, paddedAngles, stripeWidth, angleShift, nullptr)
        || cudaDeviceSynchronize() != cudaSuccess)
    {
        std::cerr << "Stripe FW CUDA smoke failed after FFT\n";
        return 25;
    }
    std::vector<float> stripeOut(stripeInputCount);
    if (cudaMemcpy(stripeOut.data(), stripeDevice, stripeInputCount * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to download stripe smoke output\n";
        return 26;
    }
    for (float value : stripeOut)
    {
        if (!std::isfinite(value))
        {
            std::cerr << "Stripe smoke output contained non-finite value\n";
            return 27;
        }
    }
    if (r2c != 0) { cufftDestroy(r2c); }
    if (c2r != 0) { cufftDestroy(c2r); }
    cudaFree(stripeDevice);
    cudaFree(paddedDevice);
    cudaFree(tempA);
    cudaFree(tempD);
    cudaFree(approx);
    cudaFree(cV);
    cudaFree(cH);
    cudaFree(cD);
    cudaFree(packed);
    cudaFree(spectrum);

    std::cout << "ct_workstation_tests passed\n";
    return 0;
}
