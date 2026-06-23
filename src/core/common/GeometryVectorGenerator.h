#pragma once

#include "core/config/AppConfig.h"

#include <vector>

namespace ct
{

class GeometryVectorGenerator
{
public:
    static std::vector<double> createParallel3dVectors(const GeometryConfig& geometry);
};

} // namespace ct
