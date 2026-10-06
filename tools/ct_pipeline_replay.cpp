#include "core/acq/CtTiffWriter.h"
#include "core/io/ImageFileIo.h"
#include "core/config/AppConfig.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSet>
#include <QThread>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

constexpr int kDefaultWriterThreads = 24;
constexpr int kDefaultQueueCapacity = 2048;
constexpr int kDefaultFrameIntervalMs = 10;

struct ReplaySpec
{
    QString sourceJob;
    QString targetProjectionDir;
    QString targetOutputDir;
};

struct ReplayOptions
{
    int writerThreads = kDefaultWriterThreads;
    int queueCapacity = kDefaultQueueCapacity;
    int frameIntervalMs = kDefaultFrameIntervalMs;
    bool maxSpeed = false;
    bool allowNonEmptyTargets = false;
    bool skipQueueJobs = false;
    QString queueRoot;
    QVector<ReplaySpec> specs;
};

struct SourceDataset
{
    QJsonObject job;
    QJsonObject manifest;
    QString projectionDir;
    QString outputDir;
    QString tomoPrefix;
    QString flatPrefix;
    QString darkPrefix;
    QString inputExt;
    QString filePattern0;
    QString filePattern180;
    int inputPadding = 4;
    int projectionCount = 0;
    int imageWidth = 0;
    int imageHeight = 0;
    bool autoCor = false;
    bool corReferenceCaptured = false;
    QJsonArray flatGroups;
};

struct SavePlanItem
{
    QString sourcePath;
    QString targetPath;
};

bool readJsonFile(const QString& path, QJsonObject* object, QString* errorMessage = nullptr)
{
    if (object == nullptr)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Output JSON pointer is null.");
        }
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Failed to open JSON: %1").arg(path);
        }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Invalid JSON %1: %2").arg(path, parseError.errorString());
        }
        return false;
    }
    *object = doc.object();
    return true;
}

bool writeJsonFileAtomic(const QString& path, const QJsonObject& object)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    const QString tmpPath = path + QStringLiteral(".tmp");
    QFile file(tmpPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    file.close();
    QFile::remove(path);
    return QFile::rename(tmpPath, path);
}

QString sanitizeQueueName(QString value)
{
    value = value.trimmed();
    if (value.isEmpty())
    {
        return QStringLiteral("ct_job");
    }
    for (QChar& ch : value)
    {
        if (!ch.isLetterOrNumber() && ch != QLatin1Char('_') && ch != QLatin1Char('-'))
        {
            ch = QLatin1Char('_');
        }
    }
    return value.left(80);
}

QString queueFileName(int queueIndex, const QString& baseName)
{
    return QStringLiteral("%1_%2.json")
        .arg(std::max(0, queueIndex), 6, 10, QLatin1Char('0'))
        .arg(sanitizeQueueName(baseName));
}

int queueIndexFromFileName(const QString& fileName)
{
    const int underscore = fileName.indexOf(QLatin1Char('_'));
    if (underscore <= 0)
    {
        return std::numeric_limits<int>::max();
    }
    bool ok = false;
    const int index = fileName.left(underscore).toInt(&ok);
    return ok ? index : std::numeric_limits<int>::max();
}

QString numberedFileName(const QString& prefix, int padding, int index, const QString& ext)
{
    QString normalizedExt = ext;
    if (normalizedExt.startsWith(QLatin1Char('.')))
    {
        normalizedExt.remove(0, 1);
    }
    return QStringLiteral("%1%2.%3")
        .arg(prefix)
        .arg(std::max(0, index), std::clamp(padding, 1, 12), 10, QLatin1Char('0'))
        .arg(normalizedExt);
}

QString projectionDirFromJob(const QJsonObject& job)
{
    QString projectionDir = job.value(QStringLiteral("PROJECTION_DIR")).toString();
    if (!projectionDir.isEmpty())
    {
        return QDir::fromNativeSeparators(projectionDir);
    }
    const QString manifestPath = job.value(QStringLiteral("MANIFEST_PATH")).toString();
    if (!manifestPath.isEmpty())
    {
        return QDir::fromNativeSeparators(QFileInfo(manifestPath).absolutePath());
    }
    return {};
}

QJsonObject appConfigToJobObject(const ct::AppConfig& cfg)
{
    QJsonObject job;
    job["RECON_METHOD"] = ct::toString(cfg.io.reconMethod);

    QJsonObject geometry;
    geometry["COR_SHIFT"] = cfg.geometry.corShift;
    geometry["COR_AUTO"] = cfg.geometry.corAuto;
    job["GEOMETRY"] = geometry;

    QJsonObject phase;
    phase["ENABLE"] = cfg.phase.enable;
    phase["ENERGY_KEV"] = cfg.phase.energyKev;
    phase["DISTANCE_M"] = cfg.phase.distanceM;
    phase["PIXEL_SIZE_UM"] = cfg.phase.pixelSizeUm;
    phase["ALPHA_RATIO"] = cfg.phase.alphaRatio;
    job["PHASE"] = phase;

    QJsonObject stripe;
    stripe["ENABLE"] = cfg.stripe.enable;
    stripe["METHOD"] = cfg.stripe.method.isEmpty() ? QStringLiteral("FW") : cfg.stripe.method;
    stripe["LEVEL"] = cfg.stripe.level;
    stripe["WNAME"] = cfg.stripe.waveletName.isEmpty() ? QStringLiteral("db5") : cfg.stripe.waveletName;
    stripe["SIGMA"] = cfg.stripe.sigma;
    stripe["PAD"] = cfg.stripe.pad;
    stripe["NCHUNK"] = cfg.stripe.chunk;
    job["STRIPE"] = stripe;

    QJsonObject preprocess;
    preprocess["TAPER_WIDTH"] = cfg.preprocess.taperWidth;
    preprocess["FILTER_TYPE"] = ct::toString(cfg.preprocess.filterType);
    preprocess["FLAT_GAP"] = cfg.preprocess.flatGap;
    preprocess["FLAT_NUM"] = cfg.preprocess.flatNum;
    if (!cfg.preprocess.flatGroups.isEmpty())
    {
        QJsonArray groups;
        for (const ct::FlatGroupConfig& group : cfg.preprocess.flatGroups)
        {
            QJsonObject obj;
            obj["LABEL"] = group.label;
            obj["AFTER_PROJECTION_COUNT"] = group.afterProjectionCount;
            obj["FIRST_FLAT_INDEX"] = group.firstFlatIndex;
            obj["FLAT_COUNT"] = group.flatCount;
            groups.append(obj);
        }
        preprocess["FLAT_GROUPS"] = groups;
    }
    job["PREPROCESS"] = preprocess;

    QJsonObject compute;
    compute["BATCH_SIZE"] = cfg.compute.batchSize;
    compute["Z_CHUNK_SIZE"] = cfg.compute.zChunkSize;
    compute["GPU_INDEX"] = cfg.compute.gpuIndex;
    compute["Z_BLOCK_SPLIT"] = cfg.compute.zBlockSplit;
    job["COMPUTE"] = compute;

    QJsonObject io;
    io["CORRECTION_MODE"] = ct::toString(cfg.io.correctionMode);
    io["DARK_PREFIX"] = cfg.io.darkPrefix;
    io["FILE_PREFIX"] = cfg.io.filePrefix;
    io["SAVE_DEPTH"] = ct::toString(cfg.io.saveDepth);
    io["SAVE_FMT"] = cfg.io.saveFormat;
    io["TIFF_THREADS"] = cfg.io.tiffThreads;
    job["IO"] = io;

    return job;
}

void applyJobObjectToConfig(const QJsonObject& job, ct::AppConfig& cfg)
{
    cfg.io.reconMethod = ct::reconMethodFromString(job.value(QStringLiteral("RECON_METHOD")).toString(ct::toString(cfg.io.reconMethod)));

    const QJsonObject geometry = job.value(QStringLiteral("GEOMETRY")).toObject();
    cfg.geometry.corShift = static_cast<float>(geometry.value(QStringLiteral("COR_SHIFT")).toDouble(cfg.geometry.corShift));
    cfg.geometry.corAuto = geometry.value(QStringLiteral("COR_AUTO")).toBool(cfg.geometry.corAuto);

    const QJsonObject phase = job.value(QStringLiteral("PHASE")).toObject();
    cfg.phase.enable = phase.value(QStringLiteral("ENABLE")).toBool(cfg.phase.enable);
    cfg.phase.energyKev = static_cast<float>(phase.value(QStringLiteral("ENERGY_KEV")).toDouble(cfg.phase.energyKev));
    cfg.phase.distanceM = static_cast<float>(phase.value(QStringLiteral("DISTANCE_M")).toDouble(cfg.phase.distanceM));
    cfg.phase.pixelSizeUm = static_cast<float>(phase.value(QStringLiteral("PIXEL_SIZE_UM")).toDouble(cfg.phase.pixelSizeUm));
    cfg.phase.alphaRatio = static_cast<float>(phase.value(QStringLiteral("ALPHA_RATIO")).toDouble(cfg.phase.alphaRatio));

    const QJsonObject stripe = job.value(QStringLiteral("STRIPE")).toObject();
    if (!stripe.isEmpty())
    {
        cfg.stripe.enable = stripe.value(QStringLiteral("ENABLE")).toBool(cfg.stripe.enable);
        cfg.stripe.method = stripe.value(QStringLiteral("METHOD")).toString(cfg.stripe.method).toUpper();
        if (cfg.stripe.method.isEmpty())
        {
            cfg.stripe.method = QStringLiteral("FW");
        }
        cfg.stripe.level = std::clamp(stripe.value(QStringLiteral("LEVEL")).toInt(cfg.stripe.level), 0, 20);
        cfg.stripe.waveletName = stripe.value(QStringLiteral("WNAME")).toString(cfg.stripe.waveletName);
        if (cfg.stripe.waveletName.isEmpty())
        {
            cfg.stripe.waveletName = QStringLiteral("db5");
        }
        cfg.stripe.sigma = std::max(0.01f, static_cast<float>(stripe.value(QStringLiteral("SIGMA")).toDouble(cfg.stripe.sigma)));
        cfg.stripe.pad = stripe.value(QStringLiteral("PAD")).toBool(cfg.stripe.pad);
        cfg.stripe.chunk = std::clamp(stripe.value(QStringLiteral("NCHUNK")).toInt(cfg.stripe.chunk), 1, 1024);
    }

    const QJsonObject preprocess = job.value(QStringLiteral("PREPROCESS")).toObject();
    cfg.preprocess.taperWidth = preprocess.value(QStringLiteral("TAPER_WIDTH")).toInt(cfg.preprocess.taperWidth);
    cfg.preprocess.filterType = ct::filterTypeFromString(preprocess.value(QStringLiteral("FILTER_TYPE")).toString(ct::toString(cfg.preprocess.filterType)));
    cfg.preprocess.flatGap = std::max(0, preprocess.value(QStringLiteral("FLAT_GAP")).toInt(cfg.preprocess.flatGap));
    cfg.preprocess.flatNum = std::max(1, preprocess.value(QStringLiteral("FLAT_NUM")).toInt(cfg.preprocess.flatNum));
    cfg.preprocess.flatGroups.clear();
    for (const QJsonValue& value : preprocess.value(QStringLiteral("FLAT_GROUPS")).toArray())
    {
        const QJsonObject obj = value.toObject();
        ct::FlatGroupConfig group;
        group.label = obj.value(QStringLiteral("LABEL")).toString();
        group.afterProjectionCount = obj.value(QStringLiteral("AFTER_PROJECTION_COUNT")).toInt();
        group.firstFlatIndex = obj.value(QStringLiteral("FIRST_FLAT_INDEX")).toInt();
        group.flatCount = obj.value(QStringLiteral("FLAT_COUNT")).toInt();
        if (group.flatCount > 0)
        {
            cfg.preprocess.flatGroups.append(group);
        }
    }

    const QJsonObject compute = job.value(QStringLiteral("COMPUTE")).toObject();
    cfg.compute.batchSize = compute.value(QStringLiteral("BATCH_SIZE")).toInt(cfg.compute.batchSize);
    cfg.compute.zChunkSize = compute.value(QStringLiteral("Z_CHUNK_SIZE")).toInt(cfg.compute.zChunkSize);
    cfg.compute.gpuIndex = compute.value(QStringLiteral("GPU_INDEX")).toInt(cfg.compute.gpuIndex);
    cfg.compute.zBlockSplit = compute.value(QStringLiteral("Z_BLOCK_SPLIT")).toInt(cfg.compute.zBlockSplit);

    const QJsonObject io = job.value(QStringLiteral("IO")).toObject();
    cfg.io.correctionMode = ct::correctionModeFromString(io.value(QStringLiteral("CORRECTION_MODE")).toString(ct::toString(cfg.io.correctionMode)));
    cfg.io.darkPrefix = io.value(QStringLiteral("DARK_PREFIX")).toString(cfg.io.darkPrefix);
    cfg.io.filePrefix = io.value(QStringLiteral("FILE_PREFIX")).toString(cfg.io.filePrefix);
    cfg.io.saveDepth = ct::saveDepthFromString(io.value(QStringLiteral("SAVE_DEPTH")).toString(ct::toString(cfg.io.saveDepth)));
    cfg.io.saveFormat = io.value(QStringLiteral("SAVE_FMT")).toString(cfg.io.saveFormat);
    cfg.io.tiffThreads = std::clamp(io.value(QStringLiteral("TIFF_THREADS")).toInt(cfg.io.tiffThreads), 1, 64);
}

QString inferRepoRoot()
{
    QDir dir(QCoreApplication::applicationDirPath());
    for (int i = 0; i < 8; ++i)
    {
        if (QFileInfo::exists(dir.filePath(QStringLiteral("CMakeLists.txt")))
            && QFileInfo::exists(dir.filePath(QStringLiteral(".git"))))
        {
            return dir.absolutePath();
        }
        if (!dir.cdUp())
        {
            break;
        }
    }
    return QDir::currentPath();
}

bool isDirectoryEmpty(const QString& path)
{
    QDir dir(path);
    if (!dir.exists())
    {
        return true;
    }
    return dir.entryList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System).isEmpty();
}

SourceDataset loadSourceDataset(const QString& sourceJobPath)
{
    SourceDataset dataset;
    QString error;
    if (!readJsonFile(sourceJobPath, &dataset.job, &error))
    {
        throw std::runtime_error(error.toStdString());
    }
    dataset.projectionDir = projectionDirFromJob(dataset.job);
    dataset.outputDir = QDir::fromNativeSeparators(dataset.job.value(QStringLiteral("OUTPUT_DIR")).toString());
    if (dataset.projectionDir.isEmpty())
    {
        throw std::runtime_error("source job has no PROJECTION_DIR");
    }
    const QString manifestPath = QDir(dataset.projectionDir).filePath(QStringLiteral("scan_manifest.json"));
    if (!readJsonFile(manifestPath, &dataset.manifest, &error))
    {
        throw std::runtime_error(error.toStdString());
    }
    if (!dataset.manifest.value(QStringLiteral("COMPLETED")).toBool(false))
    {
        throw std::runtime_error("source manifest is not completed");
    }

    const QJsonObject acq = dataset.manifest.value(QStringLiteral("ACQUISITION")).toObject();
    const QJsonObject io = dataset.manifest.value(QStringLiteral("IO")).toObject();
    dataset.tomoPrefix = io.value(QStringLiteral("TOMO_PREFIX")).toString(QStringLiteral("Tomo_"));
    dataset.flatPrefix = io.value(QStringLiteral("FLAT_PREFIX")).toString(QStringLiteral("Flat_"));
    dataset.darkPrefix = io.value(QStringLiteral("DARK_PREFIX")).toString(QStringLiteral("Dark_"));
    dataset.inputPadding = io.value(QStringLiteral("INPUT_PADDING")).toInt(4);
    dataset.inputExt = io.value(QStringLiteral("INPUT_EXT")).toString(QStringLiteral("tiff"));
    dataset.filePattern0 = io.value(QStringLiteral("FILE_PATTERN_0")).toString();
    dataset.filePattern180 = io.value(QStringLiteral("FILE_PATTERN_180")).toString(QStringLiteral("alignment180.tiff"));
    dataset.projectionCount = acq.value(QStringLiteral("PROJECTION_COUNT")).toInt();
    dataset.imageWidth = acq.value(QStringLiteral("IMAGE_WIDTH")).toInt();
    dataset.imageHeight = acq.value(QStringLiteral("IMAGE_HEIGHT")).toInt();
    dataset.autoCor = acq.value(QStringLiteral("AUTO_COR")).toBool(false);
    dataset.corReferenceCaptured = acq.value(QStringLiteral("COR_REFERENCE_CAPTURED")).toBool(false);
    dataset.flatGroups = dataset.manifest.value(QStringLiteral("FLAT_GROUPS")).toArray();

    if (dataset.projectionCount <= 0 || dataset.imageWidth <= 0 || dataset.imageHeight <= 0)
    {
        throw std::runtime_error("source manifest has invalid acquisition dimensions");
    }
    return dataset;
}

int tomoStartIndex(const SourceDataset& dataset)
{
    if (!dataset.filePattern0.isEmpty())
    {
        const QRegularExpression re(QStringLiteral("(\\d+)(?=\\.[^.]+$)"));
        const QRegularExpressionMatch match = re.match(dataset.filePattern0);
        if (match.hasMatch())
        {
            bool ok = false;
            const int value = match.captured(1).toInt(&ok);
            if (ok)
            {
                return value;
            }
        }
    }
    const QString zeroPath = QDir(dataset.projectionDir).filePath(numberedFileName(dataset.tomoPrefix,
        dataset.inputPadding, 0, dataset.inputExt));
    return QFileInfo::exists(zeroPath) ? 0 : 1;
}

void appendIfExistsOrThrow(QVector<SavePlanItem>& plan, const QString& sourcePath, const QString& targetPath)
{
    if (!QFileInfo::exists(sourcePath))
    {
        throw std::runtime_error(QStringLiteral("missing source frame: %1").arg(sourcePath).toStdString());
    }
    plan.append({ sourcePath, targetPath });
}

QVector<SavePlanItem> buildSavePlan(const SourceDataset& dataset, const QString& targetProjectionDir)
{
    QVector<SavePlanItem> plan;
    QSet<QString> targetPaths;
    auto appendUnique = [&](const QString& sourcePath, const QString& targetPath) {
        const QString normalizedTarget = QDir::fromNativeSeparators(QFileInfo(targetPath).absoluteFilePath());
        if (targetPaths.contains(normalizedTarget))
        {
            return;
        }
        appendIfExistsOrThrow(plan, sourcePath, targetPath);
        targetPaths.insert(normalizedTarget);
    };
    if (dataset.autoCor && dataset.corReferenceCaptured && !dataset.filePattern180.isEmpty())
    {
        appendUnique(
            QDir(dataset.projectionDir).filePath(dataset.filePattern180),
            QDir(targetProjectionDir).filePath(dataset.filePattern180));
    }

    int flatGroupCounter = 0;
    for (const QJsonValue& value : dataset.flatGroups)
    {
        const QJsonObject group = value.toObject();
        const int firstFlat = group.value(QStringLiteral("FIRST_FLAT_INDEX")).toInt();
        const int flatCount = group.value(QStringLiteral("FLAT_COUNT")).toInt();
        for (int i = 0; i < flatCount; ++i)
        {
            const QString fileName = numberedFileName(dataset.flatPrefix, dataset.inputPadding,
                firstFlat + i, dataset.inputExt);
            appendUnique(
                QDir(dataset.projectionDir).filePath(fileName),
                QDir(targetProjectionDir).filePath(fileName));
        }
        ++flatGroupCounter;
    }
    Q_UNUSED(flatGroupCounter);

    const int start = tomoStartIndex(dataset);
    for (int i = 0; i < dataset.projectionCount; ++i)
    {
        const QString fileName = numberedFileName(dataset.tomoPrefix, dataset.inputPadding,
            start + i, dataset.inputExt);
        appendUnique(
            QDir(dataset.projectionDir).filePath(fileName),
            QDir(targetProjectionDir).filePath(fileName));
    }
    if (dataset.autoCor && dataset.corReferenceCaptured && !dataset.filePattern0.isEmpty())
    {
        const QString targetRef0 = QDir(targetProjectionDir).filePath(dataset.filePattern0);
        if (!QFileInfo::exists(targetRef0))
        {
            const QString sourceRef0 = QDir(dataset.projectionDir).filePath(dataset.filePattern0);
            appendUnique(sourceRef0, targetRef0);
        }
    }
    return plan;
}

QJsonObject replayManifest(const SourceDataset& dataset)
{
    QJsonObject manifest = dataset.manifest;
    manifest["COMPLETED"] = false;
    manifest["ERROR"] = QString();
    manifest["CREATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    return manifest;
}

QJsonObject buildPendingJob(const SourceDataset& dataset, const QString& targetProjectionDir,
    const QString& targetOutputDir, int queueIndex, int writerThreads)
{
    Q_UNUSED(writerThreads);
    ct::AppConfig cfg = ct::AppConfig::defaults();
    applyJobObjectToConfig(dataset.job, cfg);
    cfg.io.reconMethod = cfg.io.reconMethod == ct::ReconMethod::RealTimeFBP ? ct::ReconMethod::OfflineFBP : cfg.io.reconMethod;
    cfg.io.watchDir = targetProjectionDir;
    cfg.io.saveDir = targetOutputDir;

    QJsonObject job = appConfigToJobObject(cfg);
    job["TYPE"] = "CT_STREAM_RECON_JOB";
    job["VERSION"] = 2;
    job["STATUS"] = "pending";
    job["QUEUE_INDEX"] = queueIndex;
    job["CREATED_AT"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    job["PROJECTION_DIR"] = targetProjectionDir;
    job["OUTPUT_DIR"] = targetOutputDir;
    job["REPLAY_SOURCE_JOB"] = QFileInfo(dataset.job.value(QStringLiteral("SOURCE_JOB_PATH")).toString()).exists()
        ? dataset.job.value(QStringLiteral("SOURCE_JOB_PATH")).toString()
        : QString();
    return job;
}

int nextQueueIndex(const QString& pendingDir)
{
    int maxIndex = 0;
    const QStringList existing = QDir(pendingDir).entryList({ QStringLiteral("*.json") }, QDir::Files, QDir::Name);
    for (const QString& fileName : existing)
    {
        maxIndex = std::max(maxIndex, queueIndexFromFileName(fileName));
    }
    if (maxIndex == std::numeric_limits<int>::max())
    {
        maxIndex = existing.size() * 10;
    }
    return maxIndex + 10;
}

void printUsage()
{
    std::cerr
        << "Usage: ct_pipeline_replay [options]\n"
        << "Options:\n"
        << "  --max-speed             Enqueue frames as fast as disk/queue allows.\n"
        << "  --frame-ms N            Sleep N ms between simulated camera frames. Default 10.\n"
        << "  --threads N             CT TIFF writer threads. Default 24.\n"
        << "  --queue N               Frame queue capacity. Default 2048.\n"
        << "  --allow-non-empty       Allow target Stack/FenMu dirs to contain files.\n"
        << "  --skip-queue-jobs       Write Stack dirs only; do not create pending jobs.\n"
        << "  --queue-root PATH       Override recon_queue root. Default app dir/recon_queue.\n";
}

ReplayOptions parseOptions(int argc, char** argv)
{
    ReplayOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const QString arg = QString::fromLocal8Bit(argv[i]);
        auto readInt = [&](int* out) {
            if (i + 1 >= argc || out == nullptr)
            {
                return false;
            }
            bool ok = false;
            const int value = QString::fromLocal8Bit(argv[++i]).toInt(&ok);
            if (ok)
            {
                *out = value;
            }
            return ok;
        };
        if (arg == QStringLiteral("--max-speed"))
        {
            options.maxSpeed = true;
            options.frameIntervalMs = 0;
        }
        else if (arg == QStringLiteral("--frame-ms"))
        {
            if (!readInt(&options.frameIntervalMs))
            {
                throw std::runtime_error("invalid --frame-ms value");
            }
            options.frameIntervalMs = std::max(0, options.frameIntervalMs);
        }
        else if (arg == QStringLiteral("--threads"))
        {
            if (!readInt(&options.writerThreads))
            {
                throw std::runtime_error("invalid --threads value");
            }
            options.writerThreads = std::clamp(options.writerThreads, 1, 64);
        }
        else if (arg == QStringLiteral("--queue"))
        {
            if (!readInt(&options.queueCapacity))
            {
                throw std::runtime_error("invalid --queue value");
            }
            options.queueCapacity = std::max(1, options.queueCapacity);
        }
        else if (arg == QStringLiteral("--allow-non-empty"))
        {
            options.allowNonEmptyTargets = true;
        }
        else if (arg == QStringLiteral("--skip-queue-jobs"))
        {
            options.skipQueueJobs = true;
        }
        else if (arg == QStringLiteral("--queue-root"))
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("missing --queue-root value");
            }
            options.queueRoot = QDir::fromNativeSeparators(QString::fromLocal8Bit(argv[++i]));
        }
        else if (arg == QStringLiteral("--help") || arg == QStringLiteral("-h"))
        {
            printUsage();
            std::exit(0);
        }
        else
        {
            throw std::runtime_error(QStringLiteral("unknown option: %1").arg(arg).toStdString());
        }
    }

    const QString repoRoot = inferRepoRoot();
    options.specs = {
        { QDir(repoRoot).filePath(QStringLiteral("build/vs2022-release/recon_queue/done/000010_32_.json")),
            QStringLiteral("F:/Stack1"), QStringLiteral("F:/FenMu") },
        { QDir(repoRoot).filePath(QStringLiteral("build/vs2022-release/recon_queue/done/000010_2-1.json")),
            QStringLiteral("F:/Stack2"), QStringLiteral("F:/FenMu2") },
        { QDir(repoRoot).filePath(QStringLiteral("build/vs2022-release/recon_queue/done/000010_test.json")),
            QStringLiteral("F:/Stack3"), QStringLiteral("F:/FenMu3") },
    };
    if (options.queueRoot.isEmpty())
    {
        options.queueRoot = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("recon_queue"));
    }
    return options;
}

void replayOne(const ReplaySpec& spec, const ReplayOptions& options)
{
    const SourceDataset dataset = loadSourceDataset(spec.sourceJob);
    const QString targetProjectionDir = QDir::fromNativeSeparators(spec.targetProjectionDir);
    const QString targetOutputDir = QDir::fromNativeSeparators(spec.targetOutputDir);
    const QVector<SavePlanItem> plan = buildSavePlan(dataset, targetProjectionDir);
    const qint64 frameBytes = static_cast<qint64>(dataset.imageWidth) * static_cast<qint64>(dataset.imageHeight) * 2;
    const double capacityGiB = static_cast<double>(options.queueCapacity) * static_cast<double>(frameBytes)
        / (1024.0 * 1024.0 * 1024.0);

    QDir().mkpath(targetProjectionDir);
    QDir().mkpath(targetOutputDir);
    const QString manifestPath = QDir(targetProjectionDir).filePath(QStringLiteral("scan_manifest.json"));
    QJsonObject manifest = replayManifest(dataset);
    if (!writeJsonFileAtomic(manifestPath, manifest))
    {
        throw std::runtime_error(QStringLiteral("failed to write manifest: %1").arg(manifestPath).toStdString());
    }

    ct::CtTiffWriter writer;
    std::atomic<int> latestQueued = 0;
    std::atomic<int> latestSaved = 0;
    QObject::connect(&writer, &ct::CtTiffWriter::logMessage, [](const QString& message) {
        std::cout << message.toLocal8Bit().constData() << "\n";
        std::cout.flush();
    });
    QObject::connect(&writer, &ct::CtTiffWriter::queueChanged, [&](int queued, int saved) {
        latestQueued = queued;
        latestSaved = saved;
    });

    bool writerOk = false;
    QString writerMessage;
    QObject::connect(&writer, &ct::CtTiffWriter::finished, [&](bool ok, const QString& message) {
        writerOk = ok;
        writerMessage = message;
    });

    std::cout << "Replay source: " << dataset.projectionDir.toLocal8Bit().constData() << "\n"
              << "Replay target: " << targetProjectionDir.toLocal8Bit().constData() << "\n"
              << "Frames: " << plan.size()
              << ", image=" << dataset.imageWidth << "x" << dataset.imageHeight
              << ", queue=" << options.queueCapacity << " frames (" << capacityGiB << " GiB)"
              << ", threads=" << options.writerThreads << "\n";
    std::cout.flush();

    writer.configure(options.writerThreads, options.queueCapacity);
    writer.start();
    QElapsedTimer timer;
    timer.start();
    int submitted = 0;
    for (const SavePlanItem& item : plan)
    {
        cv::Mat image = ct::readImageFile(item.sourcePath);
        if (image.empty())
        {
            writer.stop();
            throw std::runtime_error(QStringLiteral("failed to read source frame: %1").arg(item.sourcePath).toStdString());
        }
        if (image.type() != CV_16U)
        {
            writer.stop();
            throw std::runtime_error(QStringLiteral("source frame is not uint16: %1").arg(item.sourcePath).toStdString());
        }
        ct::CtFrameSaveItem saveItem;
        saveItem.filePath = item.targetPath;
        saveItem.ownedImage = std::make_shared<cv::Mat>(std::move(image));
        writer.enqueue(saveItem);
        ++submitted;
        if (!options.maxSpeed && options.frameIntervalMs > 0)
        {
            QThread::msleep(static_cast<unsigned long>(options.frameIntervalMs));
        }
        if (submitted % 50 == 0 || submitted == plan.size())
        {
            const double elapsed = std::max(0.001, timer.elapsed() / 1000.0);
            const double readRate = submitted / elapsed;
            std::cout << "Replay progress submitted=" << submitted << "/" << plan.size()
                      << ", queued=" << latestQueued.load()
                      << ", saved=" << latestSaved.load()
                      << ", submitRate=" << readRate << " fps\n";
            std::cout.flush();
        }
    }
    writer.finish();
    if (!writerOk)
    {
        manifest["COMPLETED"] = false;
        manifest["ERROR"] = writerMessage;
        writeJsonFileAtomic(manifestPath, manifest);
        throw std::runtime_error(writerMessage.toStdString());
    }
    if (latestSaved.load() != plan.size())
    {
        manifest["COMPLETED"] = false;
        manifest["ERROR"] = QStringLiteral("saved count mismatch: %1/%2").arg(latestSaved.load()).arg(plan.size());
        writeJsonFileAtomic(manifestPath, manifest);
        throw std::runtime_error(manifest.value(QStringLiteral("ERROR")).toString().toStdString());
    }
    manifest["COMPLETED"] = true;
    manifest["ERROR"] = QString();
    if (!writeJsonFileAtomic(manifestPath, manifest))
    {
        throw std::runtime_error(QStringLiteral("failed to finalize manifest: %1").arg(manifestPath).toStdString());
    }

    if (!options.skipQueueJobs)
    {
        const QString pendingDir = QDir(options.queueRoot).filePath(QStringLiteral("pending"));
        QDir().mkpath(pendingDir);
        QDir().mkpath(QDir(options.queueRoot).filePath(QStringLiteral("running")));
        QDir().mkpath(QDir(options.queueRoot).filePath(QStringLiteral("done")));
        QDir().mkpath(QDir(options.queueRoot).filePath(QStringLiteral("failed")));
        const int queueIndex = nextQueueIndex(pendingDir);
        QString base = sanitizeQueueName(QFileInfo(targetProjectionDir).fileName());
        if (base.isEmpty())
        {
            base = QStringLiteral("replay");
        }
        const QString jobPath = QDir(pendingDir).filePath(queueFileName(queueIndex, base));
        QJsonObject job = buildPendingJob(dataset, targetProjectionDir, targetOutputDir, queueIndex, options.writerThreads);
        if (!writeJsonFileAtomic(jobPath, job))
        {
            throw std::runtime_error(QStringLiteral("failed to write pending job: %1").arg(jobPath).toStdString());
        }
        std::cout << "Pending job written: " << jobPath.toLocal8Bit().constData() << "\n";
        std::cout.flush();
    }

    const double elapsed = std::max(0.001, timer.elapsed() / 1000.0);
    const double mib = static_cast<double>(latestSaved.load()) * static_cast<double>(frameBytes) / (1024.0 * 1024.0);
    std::cout << "Replay finished: saved=" << latestSaved.load()
              << ", elapsed=" << elapsed << " s"
              << ", writerThroughput=" << (mib / elapsed) << " MiB/s\n";
    std::cout.flush();
}

void validateTargets(const ReplayOptions& options)
{
    if (options.allowNonEmptyTargets)
    {
        return;
    }
    for (const ReplaySpec& spec : options.specs)
    {
        for (const QString& path : { spec.targetProjectionDir, spec.targetOutputDir })
        {
            if (!isDirectoryEmpty(path))
            {
                throw std::runtime_error(QStringLiteral("target directory is not empty: %1").arg(path).toStdString());
            }
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try
    {
        ReplayOptions options = parseOptions(argc, argv);
        validateTargets(options);
        std::cout << "CT pipeline replay started. maxSpeed=" << (options.maxSpeed ? "true" : "false")
                  << ", frameMs=" << options.frameIntervalMs
                  << ", threads=" << options.writerThreads
                  << ", queue=" << options.queueCapacity
                  << ", queueRoot=" << options.queueRoot.toLocal8Bit().constData() << "\n";
        std::cout.flush();
        for (const ReplaySpec& spec : options.specs)
        {
            replayOne(spec, options);
        }
        std::cout << "CT pipeline replay completed.\n";
        std::cout.flush();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "CT pipeline replay failed: " << ex.what() << "\n";
        printUsage();
        return 1;
    }
}
