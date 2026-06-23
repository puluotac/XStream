#pragma once

#include "core/config/AppConfig.h"

#include <QObject>

#include <atomic>
#include <memory>

namespace ct
{

class IReconstructionVolume;

struct SaveRequest
{
    std::shared_ptr<IReconstructionVolume> volume;
    AppConfig config;
    bool useDisplayWindow = false;
    float displayMin = 0.0f;
    float displayMax = 0.0f;
};

class SaveWorker : public QObject
{
    Q_OBJECT

public:
    explicit SaveWorker(SaveRequest request, QObject* parent = nullptr);

public slots:
    void process();
    void requestStop();

signals:
    void logMessage(const QString& text);
    void progressUpdated(int current, int total);
    void saveFinished();
    void saveFailed(const QString& error);

private:
    SaveRequest request_;
    std::atomic_bool stopRequested_ = false;
};

} // namespace ct
