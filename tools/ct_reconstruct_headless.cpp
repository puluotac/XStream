#include "core/config/AppConfig.h"
#include "core/recon/ReconstructionRuntimeCache.h"
#include "core/recon/ReconstructionWorker.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QTextStream>

#include <algorithm>
#include <iostream>
#include <memory>

namespace
{

void appendHeadlessDebug(const QString& text)
{
    QFile file(QCoreApplication::applicationDirPath() + QStringLiteral("/ct_headless_debug.log"));
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        QTextStream stream(&file);
        stream << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << " " << text << '\n';
    }
}

bool parseIntOption(int argc, char** argv, int& index, int* outValue)
{
    if (index + 1 >= argc || outValue == nullptr)
    {
        return false;
    }

    bool ok = false;
    const int value = QString::fromLocal8Bit(argv[index + 1]).toInt(&ok);
    if (!ok)
    {
        return false;
    }

    *outValue = value;
    ++index;
    return true;
}

bool parseMethodOption(int argc, char** argv, int& index, ct::ReconMethod* outValue)
{
    if (index + 1 >= argc || outValue == nullptr)
    {
        return false;
    }

    const QString value = QString::fromLocal8Bit(argv[index + 1]);
    *outValue = ct::reconMethodFromString(value);
    ++index;
    return true;
}

void printUsage()
{
    std::cerr
        << "Usage: ct_reconstruct_headless <config.json> [options]\n"
        << "Options:\n"
        << "  --angles N    Override TOTAL_ANGLES for short profiling runs.\n"
        << "  --batch N     Override BATCH_SIZE.\n"
        << "  --z N         Override Z_CHUNK_SIZE.\n"
        << "  --gpu N       Override GPU_INDEX.\n"
        << "  --method M    Override RECON_METHOD: realtime-fbp, offline-fbp, offline-gridrec.\n"
        << "  --tiff-threads N Override IO.TIFF_THREADS.\n";
}

} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    appendHeadlessDebug(QStringLiteral("main enter argc=%1").arg(argc));

    if (argc < 2)
    {
        printUsage();
        return 2;
    }

    ct::AppConfig config;
    QString error;
    appendHeadlessDebug(QStringLiteral("loading config %1").arg(QString::fromLocal8Bit(argv[1])));
    if (!ct::AppConfig::tryLoadFromJsonFile(QString::fromLocal8Bit(argv[1]), &config, &error))
    {
        std::cerr << "Failed to load config: " << error.toLocal8Bit().constData() << "\n";
        return 2;
    }

    for (int i = 2; i < argc; ++i)
    {
        const QString option = QString::fromLocal8Bit(argv[i]);
        if (option == QStringLiteral("--angles"))
        {
            if (!parseIntOption(argc, argv, i, &config.geometry.totalAngles))
            {
                std::cerr << "Invalid --angles value.\n";
                return 2;
            }
        }
        else if (option == QStringLiteral("--batch"))
        {
            if (!parseIntOption(argc, argv, i, &config.compute.batchSize))
            {
                std::cerr << "Invalid --batch value.\n";
                return 2;
            }
        }
        else if (option == QStringLiteral("--z"))
        {
            if (!parseIntOption(argc, argv, i, &config.compute.zChunkSize))
            {
                std::cerr << "Invalid --z value.\n";
                return 2;
            }
        }
        else if (option == QStringLiteral("--gpu"))
        {
            if (!parseIntOption(argc, argv, i, &config.compute.gpuIndex))
            {
                std::cerr << "Invalid --gpu value.\n";
                return 2;
            }
        }
        else if (option == QStringLiteral("--method"))
        {
            if (!parseMethodOption(argc, argv, i, &config.io.reconMethod))
            {
                std::cerr << "Invalid --method value.\n";
                return 2;
            }
        }
        else if (option == QStringLiteral("--tiff-threads"))
        {
            if (!parseIntOption(argc, argv, i, &config.io.tiffThreads))
            {
                std::cerr << "Invalid --tiff-threads value.\n";
                return 2;
            }
            config.io.tiffThreads = std::clamp(config.io.tiffThreads, 1, 64);
        }
        else
        {
            std::cerr << "Unknown option: " << option.toLocal8Bit().constData() << "\n";
            printUsage();
            return 2;
        }
    }

    config.preview.enableLive = false;
    appendHeadlessDebug(QStringLiteral("config ready method=%1 watch=%2 save=%3 angles=%4")
        .arg(ct::toString(config.io.reconMethod), config.io.watchDir, config.io.saveDir)
        .arg(config.geometry.totalAngles));

    std::shared_ptr<ct::ReconstructionRuntimeCache> runtimeCache;
    if (config.io.reconMethod != ct::ReconMethod::OfflineFBP
        && config.io.reconMethod != ct::ReconMethod::OfflineGridrec)
    {
        runtimeCache = std::make_shared<ct::ReconstructionRuntimeCache>();
    }
    appendHeadlessDebug(QStringLiteral("constructing worker"));
    ct::ReconstructionWorker worker(config, runtimeCache);
    appendHeadlessDebug(QStringLiteral("worker constructed"));
    QElapsedTimer timer;
    timer.start();
    int exitCode = 1;
    bool terminalSignalSeen = false;

    std::cout
        << "Headless config: angles=" << config.geometry.totalAngles
        << ", batch=" << config.compute.batchSize
        << ", zChunk=" << config.compute.zChunkSize
        << ", gpu=" << config.compute.gpuIndex
        << ", tiffThreads=" << config.io.tiffThreads
        << ", method=" << ct::toString(config.io.reconMethod).toLocal8Bit().constData() << "\n";
    std::cout.flush();

    QObject::connect(&worker, &ct::ReconstructionWorker::logMessage, [](const QString& message) {
        std::cout << message.toLocal8Bit().constData() << "\n";
        std::cout.flush();
    });
    QObject::connect(&worker, &ct::ReconstructionWorker::progressUpdated, [](int current, int total) {
        std::cout << "Progress " << current << "/" << total << "\n";
        std::cout.flush();
    });
    QObject::connect(&worker, &ct::ReconstructionWorker::corComputed, [](float cor) {
        std::cout << "COR " << cor << "\n";
        std::cout.flush();
    });
    QObject::connect(&worker, &ct::ReconstructionWorker::volumeGeometryResolved, [](int width, int height) {
        std::cout << "Volume " << width << "x" << width << "x" << height << "\n";
        std::cout.flush();
    });
    QObject::connect(&worker, &ct::ReconstructionWorker::reconstructionFinished, [&]() {
        std::cout << "Headless reconstruction finished in " << (timer.elapsed() / 1000.0) << " s\n";
        std::cout.flush();
        exitCode = 0;
        terminalSignalSeen = true;
    });
    QObject::connect(&worker, &ct::ReconstructionWorker::reconstructionFailed, [&](const QString& message) {
        std::cerr << "Headless reconstruction failed: " << message.toLocal8Bit().constData() << "\n";
        std::cerr.flush();
        exitCode = 1;
        terminalSignalSeen = true;
    });

    worker.process();
    if (!terminalSignalSeen)
    {
        std::cerr << "Headless reconstruction returned without finished/failed signal.\n";
        std::cerr.flush();
        return 1;
    }
    return exitCode;
}
