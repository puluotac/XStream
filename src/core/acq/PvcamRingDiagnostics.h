#pragma once

#include <QtGlobal>

namespace ct
{

struct PvcamRingEstimate
{
    int unread = 0;
    int capacity = 0;
    int peakUnread = 0;
};

class PvcamRingWaterlineTracker
{
public:
    void reset(int capacity);
    PvcamRingEstimate update(qint64 driverLatestFrame, qint64 lastRetrievedFrame);

private:
    int capacity_ = 0;
    int peakUnread_ = 0;
};

} // namespace ct
