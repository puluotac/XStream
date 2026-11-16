#pragma once

#include <QJsonObject>
#include <QString>
#include <QVector>
#include <atomic>
#include <functional>
#include <opencv2/core.hpp>

namespace ct
{

// All distances are micrometres. The reference is the centre of the sample
// at its initial, aligned X0/X90/Z position, not the lower grid corner.
struct MosaicConfig
{
    double pixelUm = 0.65;
    double extentX0Um = 10000;
    double extentX90Um = 1000;
    double extentZUm = 8000;
    double overlap = 0.25;
    double usableDiameter = 0.90;
    int settleMs = 300;
    double flatMoveUm = 0;
    int flatCount = 10;
    QString darkPath;
    // Mapping from positive MOTOR motion to object coordinates in reconstructed
    // slices at R=0. Must be calibrated on the beamline before acquisition.
    bool calibrated = false;
    double motorToReconDeg = 0;
    int x90Sign = 1;
    int zSign = -1;
    int previewMax = 128;
    int searchRadius = 3;
    double minNcc = 0.65;
    double maxResidualPx = 1;
    QJsonObject toJson() const;
    static MosaicConfig fromJson(const QJsonObject &);
};

struct MosaicTile
{
    int id = 0, ix = 0, iy = 0, iz = 0;
    cv::Vec3d motorOffset{0, 0, 0};
    cv::Vec3d voxelOrigin{0, 0, 0};
    QString directory;
    QJsonObject toJson() const;
};

struct MosaicPlan
{
    MosaicConfig config;
    int width = 0, height = 0;
    QVector<MosaicTile> tiles;
    // Rectangular core wholly inside the circular valid reconstruction area.
    double coreXUm = 0, coreYUm = 0;
    QJsonObject toJson() const;
};

MosaicPlan makeMosaicPlan(const MosaicConfig &, int width, int height);

struct MosaicEdge
{
    int a = 0, b = 0;
    cv::Vec3d delta{0, 0, 0}; // origin[b] - origin[a], reconstruction voxels
    double ncc = 0, residual = 0;
    double gain = 1, bias = 0; // intensity[a] ~= gain * intensity[b] + bias
};

using MosaicLog = std::function<void(const QString &)>;
// Input volumes: float slices in increasing Z, all dimensions identical.
// Nominal positions are physical geometry, NOT estimates from image content.
MosaicEdge registerMosaicPair(const QVector<cv::Mat> &a, const QVector<cv::Mat> &b, cv::Vec3d nominal, int searchRadius,
                              double minNcc, const std::atomic_bool *stop = nullptr);
QVector<cv::Vec3d> solveMosaicPositions(const QVector<cv::Vec3d> &nominal, QVector<MosaicEdge> &edges,
                                        double maxResidual);

// manifest: immutable acquisition plan. All input directories must contain
// recon_result.json with STATUS=done and unwindowed float32 TIFF slices.
// Results and registration are written only into a NEW output directory.
// registeredGeometry optionally freezes an earlier successful registration
// when repeating the SAME dataset with another Alpha.
void stitchMosaic(const QString &manifest, const QString &output, const QString &tileReconName,
                  const QString &registeredGeometry, std::atomic_bool &stop, const MosaicLog &log);

} // namespace ct
