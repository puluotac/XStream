#include <master.h>
#include <pvcam.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct Options
{
    int frames = 10;
    uns32 exposureMs = 10;
    int roiWidth = 512;
    int roiHeight = 512;
    bool fullFrame = false;
    std::filesystem::path outputDir = "E:/2026_Chun/2026_06_09/pvcam_capture_test";
};

std::string pvcamError()
{
    char message[ERROR_MSG_LEN] = {};
    const int16 code = pl_error_code();
    pl_error_message(code, message);
    return "PVCAM error " + std::to_string(code) + ": " + message;
}

bool check(bool ok, const std::string& action)
{
    if (ok)
    {
        return true;
    }
    std::cerr << action << " failed: " << pvcamError() << '\n';
    return false;
}

bool parseRoi(const std::string& value, int& width, int& height)
{
    const size_t xPos = value.find_first_of("xX");
    if (xPos == std::string::npos)
    {
        return false;
    }

    try
    {
        width = std::stoi(value.substr(0, xPos));
        height = std::stoi(value.substr(xPos + 1));
    }
    catch (...)
    {
        return false;
    }
    return width > 0 && height > 0;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto nextValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--frames")
        {
            options.frames = std::stoi(nextValue("--frames"));
        }
        else if (arg == "--exposure-ms")
        {
            options.exposureMs = static_cast<uns32>(std::stoul(nextValue("--exposure-ms")));
        }
        else if (arg == "--roi")
        {
            if (!parseRoi(nextValue("--roi"), options.roiWidth, options.roiHeight))
            {
                throw std::runtime_error("ROI must use WIDTHxHEIGHT format, for example 512x512.");
            }
        }
        else if (arg == "--full-frame")
        {
            options.fullFrame = true;
        }
        else if (arg == "--output")
        {
            options.outputDir = nextValue("--output");
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: pvcam_capture_test [--frames N] [--exposure-ms N] [--roi WxH]\n"
                << "                          [--full-frame] [--output DIR]\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    options.frames = std::max(1, options.frames);
    options.exposureMs = std::max<uns32>(1, options.exposureMs);
    return options;
}

template <typename T>
bool getParam(int16 hcam, uns32 id, int16 attr, T& value, const char* name, bool required = true)
{
    const bool ok = PV_OK == pl_get_param(hcam, id, attr, &value);
    if (!ok && required)
    {
        std::cerr << "pl_get_param(" << name << ") failed: " << pvcamError() << '\n';
    }
    return ok;
}

bool enumContains(int16 hcam, uns32 paramId, int32 wanted)
{
    uns32 count = 0;
    if (PV_OK != pl_get_param(hcam, paramId, ATTR_COUNT, &count))
    {
        return false;
    }

    for (uns32 index = 0; index < count; ++index)
    {
        int32 value = 0;
        uns32 length = 0;
        if (PV_OK != pl_enum_str_length(hcam, paramId, index, &length) || length == 0)
        {
            continue;
        }

        std::vector<char> text(length + 1, '\0');
        if (PV_OK == pl_get_enum_param(hcam, paramId, index, &value, text.data(), length)
            && value == wanted)
        {
            return true;
        }
    }
    return false;
}

int16 selectInternalExposureMode(int16 hcam)
{
    if (enumContains(hcam, PARAM_EXPOSURE_MODE, TIMED_MODE))
    {
        std::cout << "Selected exposure mode: TIMED_MODE\n";
        return TIMED_MODE;
    }

    if (enumContains(hcam, PARAM_EXPOSURE_MODE, EXT_TRIG_INTERNAL))
    {
        int16 exposeOut = EXPOSE_OUT_FIRST_ROW;

        uns32 exposeOutCount = 0;
        if (PV_OK == pl_get_param(hcam, PARAM_EXPOSE_OUT_MODE, ATTR_COUNT, &exposeOutCount)
            && exposeOutCount > 0)
        {
            int32 value = 0;
            uns32 length = 0;
            if (PV_OK == pl_enum_str_length(hcam, PARAM_EXPOSE_OUT_MODE, 0, &length) && length > 0)
            {
                std::vector<char> text(length + 1, '\0');
                if (PV_OK == pl_get_enum_param(hcam, PARAM_EXPOSE_OUT_MODE, 0, &value, text.data(), length))
                {
                    exposeOut = static_cast<int16>(value);
                }
            }
        }

        std::cout << "Selected exposure mode: EXT_TRIG_INTERNAL | exposeOut " << exposeOut << '\n';
        return static_cast<int16>(EXT_TRIG_INTERNAL | exposeOut);
    }

    throw std::runtime_error("Camera does not report EXT_TRIG_INTERNAL or TIMED_MODE exposure support.");
}

const char* exposureResolutionName(int32 expRes)
{
    switch (expRes)
    {
    case EXP_RES_ONE_MILLISEC:
        return "milliseconds";
    case EXP_RES_ONE_MICROSEC:
        return "microseconds";
    case EXP_RES_ONE_SEC:
        return "seconds";
    default:
        return "unknown";
    }
}

struct ExposureResolution
{
    int32 mode = EXP_RES_ONE_MILLISEC;
    double unitsPerMillisecond = 1.0;
};

ExposureResolution configureExposureResolution(int16 hcam)
{
    ExposureResolution result;

    int32 current = EXP_RES_ONE_MILLISEC;
    const bool hasCurrent = getParam(hcam, PARAM_EXP_RES, ATTR_CURRENT, current, "PARAM_EXP_RES", false);

    uns32 count = 0;
    if (PV_OK == pl_get_param(hcam, PARAM_EXP_RES, ATTR_COUNT, &count))
    {
        for (uns32 index = 0; index < count; ++index)
        {
            int32 value = 0;
            uns32 length = 0;
            if (PV_OK != pl_enum_str_length(hcam, PARAM_EXP_RES, index, &length) || length == 0)
            {
                continue;
            }

            std::vector<char> text(length + 1, '\0');
            if (PV_OK == pl_get_enum_param(hcam, PARAM_EXP_RES, index, &value, text.data(), length))
            {
                std::cout << "Exposure resolution option " << index << ": value=" << value
                          << ", text=" << text.data() << '\n';
                if (value == EXP_RES_ONE_MILLISEC)
                {
                    uns16 expResIndex = static_cast<uns16>(index);
                    if (PV_OK == pl_set_param(hcam, PARAM_EXP_RES_INDEX, &expResIndex))
                    {
                        current = EXP_RES_ONE_MILLISEC;
                        std::cout << "Selected exposure resolution: milliseconds\n";
                    }
                    else
                    {
                        std::cout << "Failed to set millisecond exposure resolution: " << pvcamError() << '\n';
                    }
                    break;
                }
            }
        }
    }
    else if (hasCurrent)
    {
        std::cout << "Exposure resolution options unavailable; current is " << exposureResolutionName(current) << '\n';
    }
    else
    {
        std::cout << "Exposure resolution unavailable; assuming milliseconds.\n";
    }

    result.mode = current;
    switch (current)
    {
    case EXP_RES_ONE_MICROSEC:
        result.unitsPerMillisecond = 1000.0;
        break;
    case EXP_RES_ONE_SEC:
        result.unitsPerMillisecond = 0.001;
        break;
    case EXP_RES_ONE_MILLISEC:
    default:
        result.unitsPerMillisecond = 1.0;
        break;
    }
    std::cout << "Exposure resolution in use: " << exposureResolutionName(current)
              << ", units/ms=" << result.unitsPerMillisecond << '\n';
    return result;
}

rgn_type makeCenteredRoi(uns16 sensorWidth, uns16 sensorHeight, const Options& options)
{
    const int roiWidth = options.fullFrame ? sensorWidth : std::min<int>(options.roiWidth, sensorWidth);
    const int roiHeight = options.fullFrame ? sensorHeight : std::min<int>(options.roiHeight, sensorHeight);
    const int x0 = (static_cast<int>(sensorWidth) - roiWidth) / 2;
    const int y0 = (static_cast<int>(sensorHeight) - roiHeight) / 2;

    rgn_type roi{};
    roi.s1 = static_cast<uns16>(x0);
    roi.s2 = static_cast<uns16>(x0 + roiWidth - 1);
    roi.sbin = 1;
    roi.p1 = static_cast<uns16>(y0);
    roi.p2 = static_cast<uns16>(y0 + roiHeight - 1);
    roi.pbin = 1;
    return roi;
}

int roiWidth(const rgn_type& roi)
{
    return (roi.s2 - roi.s1 + 1) / roi.sbin;
}

int roiHeight(const rgn_type& roi)
{
    return (roi.p2 - roi.p1 + 1) / roi.pbin;
}

int cvTypeForImageFormat(int32 imageFormat)
{
    switch (imageFormat)
    {
    case PL_IMAGE_FORMAT_MONO8:
    case PL_IMAGE_FORMAT_BAYER8:
        return CV_8UC1;
    case PL_IMAGE_FORMAT_MONO16:
    case PL_IMAGE_FORMAT_BAYER16:
        return CV_16UC1;
    case PL_IMAGE_FORMAT_MONO32:
    case PL_IMAGE_FORMAT_BAYER32:
        return CV_32SC1;
    default:
        throw std::runtime_error("Unsupported PVCAM image format: " + std::to_string(imageFormat));
    }
}

bool waitForReadout(int16 hcam, uns32 timeoutMs, uns32& bytesArrived)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    int16 status = 0;
    int16 lastStatus = -1;
    bytesArrived = 0;

    while (std::chrono::steady_clock::now() < deadline)
    {
        if (PV_OK != pl_exp_check_status(hcam, &status, &bytesArrived))
        {
            std::cerr << "pl_exp_check_status failed: " << pvcamError() << '\n';
            return false;
        }

        if (status == READOUT_COMPLETE || status == FRAME_AVAILABLE)
        {
            return true;
        }

        if (status != EXPOSURE_IN_PROGRESS && status != READOUT_IN_PROGRESS)
        {
            std::cerr << "Unexpected acquisition status: " << status << '\n';
            return false;
        }

        lastStatus = status;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cerr << "Timed out waiting for frame readout. Last status=" << lastStatus
              << ", bytes arrived=" << bytesArrived << '\n';
    return false;
}

std::filesystem::path framePath(const std::filesystem::path& dir, int index)
{
    std::ostringstream name;
    name << "kinetix_frame_" << std::setw(4) << std::setfill('0') << index << ".tiff";
    return dir / name.str();
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    try
    {
        options = parseOptions(argc, argv);
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 2;
    }

    int16 hcam = -1;
    bool initialized = false;
    bool opened = false;
    bool acquisitionSetup = false;

    try
    {
        std::filesystem::create_directories(options.outputDir);

        if (!check(PV_OK == pl_pvcam_init(), "pl_pvcam_init"))
        {
            return 1;
        }
        initialized = true;

        int16 total = 0;
        if (!check(PV_OK == pl_cam_get_total(&total), "pl_cam_get_total") || total <= 0)
        {
            throw std::runtime_error("No PVCAM cameras found.");
        }

        char cameraName[CAM_NAME_LEN] = {};
        if (!check(PV_OK == pl_cam_get_name(0, cameraName), "pl_cam_get_name(0)"))
        {
            return 1;
        }

        if (!check(PV_OK == pl_cam_open(cameraName, &hcam, OPEN_EXCLUSIVE), "pl_cam_open"))
        {
            return 1;
        }
        opened = true;
        std::cout << "Opened camera: " << cameraName << '\n';

        uns16 sensorWidth = 0;
        uns16 sensorHeight = 0;
        if (!getParam(hcam, PARAM_SER_SIZE, ATTR_CURRENT, sensorWidth, "PARAM_SER_SIZE")
            || !getParam(hcam, PARAM_PAR_SIZE, ATTR_CURRENT, sensorHeight, "PARAM_PAR_SIZE"))
        {
            throw std::runtime_error("Failed to read sensor size.");
        }

        const rgn_type roi = makeCenteredRoi(sensorWidth, sensorHeight, options);
        const int width = roiWidth(roi);
        const int height = roiHeight(roi);
        const ExposureResolution expResolution = configureExposureResolution(hcam);
        const uns32 exposureUnits = std::max<uns32>(
            1,
            static_cast<uns32>(std::llround(static_cast<double>(options.exposureMs) * expResolution.unitsPerMillisecond)));
        const int16 expMode = selectInternalExposureMode(hcam);

        std::cout << "Sensor: " << sensorWidth << "x" << sensorHeight << '\n';
        std::cout << "ROI: s=[" << roi.s1 << "," << roi.s2 << "], p=[" << roi.p1 << "," << roi.p2
                  << "], output=" << width << "x" << height << '\n';
        std::cout << "Frames: " << options.frames << ", exposure: " << options.exposureMs
                  << " ms (" << exposureUnits << " PVCAM units)\n";
        std::cout << "Output: " << options.outputDir.string() << '\n';

        uns32 exposureBytes = 0;
        if (!check(PV_OK == pl_exp_setup_seq(hcam, 1, 1, const_cast<rgn_type*>(&roi), expMode,
                       exposureUnits, &exposureBytes),
                "pl_exp_setup_seq"))
        {
            return 1;
        }
        acquisitionSetup = true;

        int32 imageFormat = PL_IMAGE_FORMAT_MONO16;
        getParam(hcam, PARAM_IMAGE_FORMAT, ATTR_CURRENT, imageFormat, "PARAM_IMAGE_FORMAT", false);
        int16 bitDepth = 0;
        getParam(hcam, PARAM_BIT_DEPTH, ATTR_CURRENT, bitDepth, "PARAM_BIT_DEPTH", false);
        const int cvType = cvTypeForImageFormat(imageFormat);
        const size_t expectedImageBytes = static_cast<size_t>(width) * height * CV_ELEM_SIZE(cvType);

        std::cout << "Frame bytes from PVCAM: " << exposureBytes << '\n';
        std::cout << "Image format: " << imageFormat << ", bit depth: " << bitDepth << '\n';
        if (exposureBytes < expectedImageBytes)
        {
            throw std::runtime_error("PVCAM returned fewer bytes than the selected ROI/image format requires.");
        }

        std::vector<uns8> frameBuffer(exposureBytes);

        for (int frameIndex = 0; frameIndex < options.frames; ++frameIndex)
        {
            if (!check(PV_OK == pl_exp_start_seq(hcam, frameBuffer.data()), "pl_exp_start_seq"))
            {
                return 1;
            }

            uns32 bytesArrived = 0;
            const uns32 timeoutMs = std::max<uns32>(5000, options.exposureMs + 5000);
            if (!waitForReadout(hcam, timeoutMs, bytesArrived))
            {
                pl_exp_abort(hcam, CCS_HALT);
                return 1;
            }

            cv::Mat image(height, width, cvType, frameBuffer.data());
            const std::filesystem::path outPath = framePath(options.outputDir, frameIndex);
            if (!cv::imwrite(outPath.string(), image))
            {
                throw std::runtime_error("Failed to save image: " + outPath.string());
            }

            if (PV_OK != pl_exp_finish_seq(hcam, frameBuffer.data(), 0))
            {
                std::cerr << "pl_exp_finish_seq warning: " << pvcamError() << '\n';
            }

            std::cout << "Saved frame " << (frameIndex + 1) << "/" << options.frames
                      << ": " << outPath.string() << " (" << bytesArrived << " bytes arrived)\n";
        }

        acquisitionSetup = false;
        if (!check(PV_OK == pl_exp_uninit_seq(), "pl_exp_uninit_seq"))
        {
            return 1;
        }

        opened = false;
        if (!check(PV_OK == pl_cam_close(hcam), "pl_cam_close"))
        {
            return 1;
        }
        hcam = -1;

        initialized = false;
        if (!check(PV_OK == pl_pvcam_uninit(), "pl_pvcam_uninit"))
        {
            return 1;
        }

        std::cout << "PVCAM capture test passed.\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Capture test failed: " << ex.what() << '\n';
    }

    if (opened)
    {
        if (acquisitionSetup)
        {
            pl_exp_abort(hcam, CCS_HALT);
            pl_exp_uninit_seq();
        }
        pl_cam_close(hcam);
    }
    if (initialized)
    {
        pl_pvcam_uninit();
    }
    return 1;
}
