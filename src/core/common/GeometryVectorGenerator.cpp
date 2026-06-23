#include "core/common/GeometryVectorGenerator.h"

#include <cmath>

namespace ct
{

std::vector<double> GeometryVectorGenerator::createParallel3dVectors(const GeometryConfig& geometry)
{
    constexpr double pi = 3.14159265358979323846;
    std::vector<double> vectors(static_cast<size_t>(geometry.totalAngles) * 12, 0.0);

    for (int i = 0; i < geometry.totalAngles; ++i)
    {
        const double angle = static_cast<double>(i) * (pi / static_cast<double>(geometry.totalAngles))
            + static_cast<double>(geometry.angleOffsetDeg) * pi / 180.0;
        const double rx = std::sin(angle);
        const double ry = -std::cos(angle);
        const double ux = std::cos(angle);
        const double uy = std::sin(angle);

        const size_t base = static_cast<size_t>(i) * 12;
        vectors[base + 0] = rx;
        vectors[base + 1] = ry;
        vectors[base + 2] = 0.0;
        vectors[base + 3] = static_cast<double>(geometry.corShift) * ux;
        vectors[base + 4] = static_cast<double>(geometry.corShift) * uy;
        vectors[base + 5] = 0.0;
        vectors[base + 6] = ux;
        vectors[base + 7] = uy;
        vectors[base + 8] = 0.0;
        vectors[base + 9] = 0.0;
        vectors[base + 10] = 0.0;
        vectors[base + 11] = 1.0;
    }

    return vectors;
}

} // namespace ct
