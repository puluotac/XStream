#pragma once

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QVector>

namespace ct
{

enum class CorrectionMode
{
    None,
    Flat,
    FlatDark
};

enum class PreprocessMethod
{
    Linear,
    Log
};

enum class FilterType
{
    RamLak,
    SheppLogan,
    Cosine,
    Hamming,
    Hann
};

enum class SaveDepth
{
    Bit8,
    Bit16,
    Bit32
};

enum class ReconMethod
{
    RealTimeFBP,
    OfflineFBP,
    OfflineGridrec
};

struct GeometryConfig
{
    int volW = 1024;
    int volH = 1024;
    int totalAngles = 1200;
    float corShift = 0.0f;
    bool corAuto = true;
    float angleOffsetDeg = 0.0f;
};

struct PhaseConfig
{
    bool enable = false;
    float energyKev = 0.0f;
    float distanceM = 0.0f;
    float pixelSizeUm = 0.0f;
    float alphaRatio = 0.0f;
};

struct StripeConfig
{
    bool enable = false;
    QString method = "FW";
    int level = 0;
    QString waveletName = "db5";
    float sigma = 2.0f;
    bool pad = true;
    int chunk = 16;
};

struct FlatGroupConfig
{
    QString label;
    int afterProjectionCount = 0;
    int firstFlatIndex = 0;
    int flatCount = 0;
};

struct PreprocessConfig
{
    PreprocessMethod method = PreprocessMethod::Log;
    float airValue = 1.0f;
    int taperWidth = 0;
    FilterType filterType = FilterType::SheppLogan;
    int flatGap = 0;
    int flatNum = 1;
    QVector<FlatGroupConfig> flatGroups;
};

struct ComputeConfig
{
    int batchSize = 20;
    int zChunkSize = 64;
    int gpuIndex = 0;
    int zBlockSplit = 1;
};

struct PreviewConfig
{
    bool enableLive = true;
    int liveSliceIndex = 512;
};

struct IoConfig
{
    QString watchDir;
    QString saveDir;
    ReconMethod reconMethod = ReconMethod::RealTimeFBP;
    CorrectionMode correctionMode = CorrectionMode::FlatDark;
    QString tomoPrefix = "Tomo_";
    QString flatPrefix = "Flat_";
    QString darkPrefix = "Dark_";
    int inputPadding = 4;
    QString inputExt = "tiff";
    QString filePattern0 = "Tomo_0001.tiff";
    QString filePattern180 = "alignment180.tiff";
    QString filePrefix = "Slice";
    SaveDepth saveDepth = SaveDepth::Bit16;
    QString saveFormat = "tiff";
    int tiffThreads = 16;
};

struct MotorAxisConfig
{
    bool hasBaseSpeed = false;
    double baseSpeed = 0.0;
    double speed = 0.0;
    double accel = 0.0;
};

struct AppConfig
{
    GeometryConfig geometry;
    PhaseConfig phase;
    StripeConfig stripe;
    PreprocessConfig preprocess;
    ComputeConfig compute;
    PreviewConfig preview;
    IoConfig io;
    QMap<QString, MotorAxisConfig> motorAxes;

    static AppConfig defaults();
    static AppConfig fromJsonObject(const QJsonObject& object);
    static bool tryLoadFromJsonFile(const QString& filePath, AppConfig* outConfig, QString* errorMessage = nullptr);
    static AppConfig loadFromJsonFile(const QString& filePath);
    QJsonObject toJsonObject() const;
    bool saveToJsonFile(const QString& filePath) const;
};

QString toString(CorrectionMode mode);
QString toString(PreprocessMethod method);
QString toString(FilterType type);
QString toString(SaveDepth depth);
QString toString(ReconMethod method);

CorrectionMode correctionModeFromString(const QString& value);
PreprocessMethod preprocessMethodFromString(const QString& value);
FilterType filterTypeFromString(const QString& value);
SaveDepth saveDepthFromString(const QString& value);
ReconMethod reconMethodFromString(const QString& value);

} // namespace ct
