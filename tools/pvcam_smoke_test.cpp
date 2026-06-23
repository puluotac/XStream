#include <master.h>
#include <pvcam.h>

#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace
{

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

template <size_t N>
bool getStringParam(int16 hcam, uns32 id, std::array<char, N>& value, const char* name, bool required = true)
{
    value.fill('\0');
    const bool ok = PV_OK == pl_get_param(hcam, id, ATTR_CURRENT, value.data());
    if (!ok && required)
    {
        std::cerr << "pl_get_param(" << name << ") failed: " << pvcamError() << '\n';
    }
    return ok;
}

std::string enumValueName(int16 hcam, uns32 id, int32 currentValue)
{
    uns32 count = 0;
    if (PV_OK != pl_get_param(hcam, id, ATTR_COUNT, &count))
    {
        return {};
    }

    for (uns32 index = 0; index < count; ++index)
    {
        int32 value = 0;
        uns32 textLength = 0;
        if (PV_OK != pl_enum_str_length(hcam, id, index, &textLength) || textLength == 0)
        {
            continue;
        }

        std::vector<char> text(textLength + 1, '\0');
        if (PV_OK == pl_get_enum_param(hcam, id, index, &value, text.data(), textLength)
            && value == currentValue)
        {
            return text.data();
        }
    }

    return {};
}

void printOptionalString(int16 hcam, uns32 id, const char* label, size_t bufferSize)
{
    std::vector<char> value(bufferSize, '\0');
    if (PV_OK == pl_get_param(hcam, id, ATTR_CURRENT, value.data()))
    {
        std::cout << label << ": " << value.data() << '\n';
    }
    else
    {
        std::cout << label << ": <unavailable> (" << pvcamError() << ")\n";
    }
}

} // namespace

int main()
{
    bool initialized = false;
    int16 hcam = -1;

    if (!check(PV_OK == pl_pvcam_init(), "pl_pvcam_init"))
    {
        return 1;
    }
    initialized = true;

    uns16 version = 0;
    if (check(PV_OK == pl_pvcam_get_ver(&version), "pl_pvcam_get_ver"))
    {
        std::cout << "PVCAM version: "
                  << ((version >> 8) & 0xFF) << '.'
                  << ((version >> 4) & 0x0F) << '.'
                  << ((version >> 0) & 0x0F) << '\n';
    }

    int16 total = 0;
    if (!check(PV_OK == pl_cam_get_total(&total), "pl_cam_get_total"))
    {
        pl_pvcam_uninit();
        return 1;
    }

    std::cout << "Camera count: " << total << '\n';
    if (total <= 0)
    {
        pl_pvcam_uninit();
        return 1;
    }

    char cameraName[CAM_NAME_LEN] = {};
    if (!check(PV_OK == pl_cam_get_name(0, cameraName), "pl_cam_get_name(0)"))
    {
        pl_pvcam_uninit();
        return 1;
    }

    std::cout << "Opening camera[0]: " << cameraName << '\n';
    if (!check(PV_OK == pl_cam_open(cameraName, &hcam, OPEN_EXCLUSIVE), "pl_cam_open"))
    {
        pl_pvcam_uninit();
        return 1;
    }

    printOptionalString(hcam, PARAM_VENDOR_NAME, "Vendor", MAX_VENDOR_NAME_LEN);
    printOptionalString(hcam, PARAM_PRODUCT_NAME, "Product", MAX_PRODUCT_NAME_LEN);
    printOptionalString(hcam, PARAM_SYSTEM_NAME, "System", MAX_SYSTEM_NAME_LEN);
    printOptionalString(hcam, PARAM_CAMERA_PART_NUMBER, "Part number", MAX_CAM_PART_NUM_LEN);
    printOptionalString(hcam, PARAM_HEAD_SER_NUM_ALPHA, "Serial", MAX_ALPHA_SER_NUM_LEN);
    printOptionalString(hcam, PARAM_CHIP_NAME, "Sensor chip", CCD_NAME_LEN);

    uns16 width = 0;
    uns16 height = 0;
    if (getParam(hcam, PARAM_SER_SIZE, ATTR_CURRENT, width, "PARAM_SER_SIZE"))
    {
        std::cout << "Sensor width: " << width << '\n';
    }
    if (getParam(hcam, PARAM_PAR_SIZE, ATTR_CURRENT, height, "PARAM_PAR_SIZE"))
    {
        std::cout << "Sensor height: " << height << '\n';
    }

    int16 bitDepth = 0;
    if (getParam(hcam, PARAM_BIT_DEPTH, ATTR_CURRENT, bitDepth, "PARAM_BIT_DEPTH", false))
    {
        std::cout << "Native bit depth: " << bitDepth << '\n';
    }
    else
    {
        std::cout << "Native bit depth: <unavailable before acquisition setup>\n";
    }

    int32 imageFormat = 0;
    if (getParam(hcam, PARAM_IMAGE_FORMAT, ATTR_CURRENT, imageFormat, "PARAM_IMAGE_FORMAT", false))
    {
        const std::string name = enumValueName(hcam, PARAM_IMAGE_FORMAT, imageFormat);
        std::cout << "Image format: " << imageFormat;
        if (!name.empty())
        {
            std::cout << " (" << name << ')';
        }
        std::cout << '\n';
    }

    int32 interfaceType = 0;
    if (getParam(hcam, PARAM_CAM_INTERFACE_TYPE, ATTR_CURRENT, interfaceType, "PARAM_CAM_INTERFACE_TYPE", false))
    {
        const std::string name = enumValueName(hcam, PARAM_CAM_INTERFACE_TYPE, interfaceType);
        std::cout << "Interface type: " << interfaceType;
        if (!name.empty())
        {
            std::cout << " (" << name << ')';
        }
        std::cout << '\n';
    }

    uns16 driverVersion = 0;
    if (getParam(hcam, PARAM_DD_VERSION, ATTR_CURRENT, driverVersion, "PARAM_DD_VERSION", false))
    {
        std::cout << "Device driver version: "
                  << ((driverVersion >> 8) & 0xFF) << '.'
                  << ((driverVersion >> 4) & 0x0F) << '.'
                  << ((driverVersion >> 0) & 0x0F) << '\n';
    }

    uns16 firmwareVersion = 0;
    if (getParam(hcam, PARAM_CAM_FW_VERSION, ATTR_CURRENT, firmwareVersion, "PARAM_CAM_FW_VERSION", false))
    {
        std::cout << "Camera firmware version: "
                  << ((firmwareVersion >> 8) & 0xFF) << '.'
                  << (firmwareVersion & 0xFF) << '\n';
    }

    bool ok = true;
    ok = check(PV_OK == pl_cam_close(hcam), "pl_cam_close") && ok;
    hcam = -1;
    ok = check(PV_OK == pl_pvcam_uninit(), "pl_pvcam_uninit") && ok;
    initialized = false;

    if (initialized)
    {
        pl_pvcam_uninit();
    }

    std::cout << "PVCAM smoke test " << (ok ? "passed" : "failed") << '\n';
    return ok ? 0 : 1;
}
