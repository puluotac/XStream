#pragma once

#include "hardware/camera/CameraFrame.h"

#include <QString>

namespace ct
{

class ContinuousFrameSink
{
public:
    virtual ~ContinuousFrameSink() = default;
    virtual bool submitContinuousFrame(const CameraFrame& frame, QString* errorMessage) = 0;
    virtual void continuousCaptureFinished(bool ok, const QString& message) = 0;
};

} // namespace ct
