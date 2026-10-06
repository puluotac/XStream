#pragma once

#include "core/acq/ContinuousTiming.h"

#include <QVector>
#include <QString>

namespace ct
{

struct TimingProfileKey
{
    QString cameraIdentity = QStringLiteral("Kinetix/PVCAM");
    QString readoutMode = QStringLiteral("16-bit Dynamic Range");
    QString triggerMode = QStringLiteral("internal_timed");
    int binningX = 1;
    int binningY = 1;
    double exposureMs = 0.0;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 0;
    int roiHeight = 0;

    QString canonical() const;
};

struct TimingProfile
{
    TimingProfileKey key;
    QString cameraName;
    QString cameraPartNumber;
    QString cameraSerial;
    int bitDepth = 0;
    int metadataVersion = 0;
    QString timingSource;
    QString readoutPort;
    QString readoutSpeed;
    QString gain;
    TimingAnalysis analysis;
    QString calibratedAt;
    QString lastUsedAt;
};

class TimingProfileCache
{
public:
    bool load(const QString& filePath, QString* errorMessage = nullptr);
    bool saveAtomic(const QString& filePath, QString* errorMessage = nullptr) const;

    const TimingProfile* find(const TimingProfileKey& key) const;
    bool markUsed(const TimingProfileKey& key, const QString& timestamp);
    void upsert(TimingProfile profile);
    void clear();
    int size() const;

private:
    QVector<TimingProfile> profiles_;
};

} // namespace ct
