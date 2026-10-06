#include "core/acq/PvcamRingDiagnostics.h"

#include <algorithm>

namespace ct
{

void PvcamRingWaterlineTracker::reset(int capacity)
{
    capacity_ = std::max(0, capacity);
    peakUnread_ = 0;
}

PvcamRingEstimate PvcamRingWaterlineTracker::update(qint64 driverLatestFrame, qint64 lastRetrievedFrame)
{
    const qint64 difference = (driverLatestFrame > 0 && lastRetrievedFrame > 0)
        ? std::max<qint64>(0, driverLatestFrame - lastRetrievedFrame)
        : 0;
    const int unread = std::clamp(static_cast<int>(std::min<qint64>(difference, capacity_)), 0, capacity_);
    peakUnread_ = std::max(peakUnread_, unread);
    return { unread, capacity_, peakUnread_ };
}

} // namespace ct
