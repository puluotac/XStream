#include "core/config/AppConfig.h"
#include "core/io/ImageFileIo.h"
#include "core/stitch/Mosaic.h"
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <thread>

namespace
{
void insist(bool b, const char *m)
{
    if (!b)
        throw std::runtime_error(m);
}
void json(QString p, QJsonObject o)
{
    QFile f(p);
    insist(f.open(QIODevice::WriteOnly), "JSON open failed");
    f.write(QJsonDocument(o).toJson());
}
QJsonObject read(QString p)
{
    QFile f(p);
    insist(f.open(QIODevice::ReadOnly), "JSON read failed");
    return QJsonDocument::fromJson(f.readAll()).object();
}
float phantom(double x, double y, double z)
{
    return float(2 + 0.4 * std::sin(.173 * x + .113 * y + .231 * z) + .31 * std::cos(.321 * x - .231 * y + .171 * z) +
                 .21 * std::sin(.417 * x + .273 * y - .317 * z) + .12 * std::cos(.611 * x - .423 * y + .231 * z));
}
QVector<cv::Mat> volume(int w, int h, cv::Vec3d pos, double gain = 1, double bias = 0)
{
    QVector<cv::Mat> v;
    for (int z = 0; z < h; ++z)
    {
        cv::Mat s(w, w, CV_32F);
        for (int y = 0; y < w; ++y)
            for (int x = 0; x < w; ++x)
                s.at<float>(y, x) = float((phantom(x + pos[0], y + pos[1], z + pos[2]) - bias) / gain);
        v.append(s);
    }
    return v;
}
void selfTest(QString root)
{
    insist(!QFileInfo::exists(root), "Self-test directory must not exist");
    insist(QDir().mkpath(root), "Cannot create self-test directory");
    ct::MosaicConfig c;
    c.calibrated = true;
    auto plan = ct::makeMosaicPlan(c, 3200, 3200);
    insist(!plan.tiles.empty(), "Empty plan");
    insist(plan.coreXUm * plan.coreXUm + plan.coreYUm * plan.coreYUm <= std::pow(3200 * .65 * .9, 2) + .01,
           "Core exceeds circular support");
    bool bad = false;
    try
    {
        auto invalid = c;
        invalid.pixelUm = 0;
        ct::makeMosaicPlan(invalid, 3200, 3200);
    }
    catch (...)
    {
        bad = true;
    }
    insist(bad, "Invalid pixel size accepted");
    auto a = volume(64, 32, {0, 0, 0});
    cv::Vec3d truth(25.375, 1.25, .375);
    auto b = volume(64, 32, truth, 1.15, .2);
    auto edge = ct::registerMosaicPair(a, b, {25, 1, 0}, 2, .8);
    insist(cv::norm(edge.delta - truth) < .4, "Subvoxel registration failed");
    auto blank = a;
    for (auto &s : blank)
        s = cv::Mat(s.size(), CV_32F, cv::Scalar(1));
    bad = false;
    try
    {
        ct::registerMosaicPair(blank, blank, {20, 0, 0}, 1, .7);
    }
    catch (...)
    {
        bad = true;
    }
    insist(bad, "Blank overlap accepted");
    QVector<cv::Vec3d> nominal{{0, 0, 0}, {30, 0, 0}, {0, 30, 0}, {30, 30, 0}};
    QVector<ct::MosaicEdge> edges;
    for (auto ij : {std::pair<int, int>{0, 1}, {0, 2}, {1, 3}, {2, 3}})
    {
        ct::MosaicEdge e;
        e.a = ij.first;
        e.b = ij.second;
        e.delta = nominal[e.b] - nominal[e.a];
        e.ncc = 1;
        edges.append(e);
    }
    auto solved = ct::solveMosaicPositions(nominal, edges, .1);
    insist(cv::norm(solved[3] - nominal[3]) < 1e-6, "Global graph failed");
    edges.last().delta[0] += 6;
    bad = false;
    try
    {
        ct::solveMosaicPositions(nominal, edges, 1);
    }
    catch (...)
    {
        bad = true;
    }
    insist(bad, "Inconsistent loop accepted");
    std::atomic_bool stopped{true};
    bad = false;
    try
    {
        ct::registerMosaicPair(a, b, truth, 1, .7, &stopped);
    }
    catch (...)
    {
        bad = true;
    }
    insist(bad, "Cancellation ignored");
    c.pixelUm = 1;
    c.previewMax = 64;
    c.searchRadius = 2;
    c.minNcc = .8;
    c.maxResidualPx = 1;
    c.usableDiameter = .98;
    QJsonObject manifest{{"TYPE", "CT_STREAM_MOSAIC"}, {"VERSION", 1},      {"COMPLETED", true},
                         {"CONFIG", c.toJson()},       {"IMAGE_WIDTH", 64}, {"IMAGE_HEIGHT", 32}};
    QJsonArray tiles;
    int id = 0;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
            {
                ct::MosaicTile t;
                t.id = ++id;
                t.ix = x;
                t.iy = y;
                t.iz = z;
                t.directory = QString("Tile_%1").arg(id, 4, 10, QLatin1Char('0'));
                t.voxelOrigin = {double(x * 30), double(y * 30), double(z * 12)};
                tiles.append(t.toJson());
                QString dir = QDir(root).filePath(t.directory + "/Recon");
                QDir().mkpath(dir);
                auto v = volume(64, 32, t.voxelOrigin, 1 + .02 * (id - 1), .01 * (id - 1));
                for (int k = 0; k < v.size(); ++k)
                {
                    QString error;
                    insist(
                        ct::writeImageFile(QDir(dir).filePath(QString("Slice_%1.tiff").arg(k, 4, 10, QLatin1Char('0'))),
                                           v[k], &error),
                        "Synthetic TIFF write failed");
                }
                json(QDir(dir).filePath("recon_result.json"), {{"STATUS", "done"}});
                json(QDir(dir).filePath("recon_params.json"),
                     {{"IO", QJsonObject{{"FILE_PREFIX", "Slice"}, {"SAVE_DEPTH", "32bit"}}}});
            }
    manifest["TILES"] = tiles;
    QString input = QDir(root).filePath("mosaic_manifest.json");
    json(input, manifest);
    std::atomic_bool stop{false};
    auto log = [](const QString &s) { std::cout << s.toStdString() << std::endl; };
    QString output = QDir(root).filePath("MosaicRecon");
    ct::stitchMosaic(input, output, "Recon", {}, stop, log);
    auto report = read(QDir(output).filePath("reconstruction.json"));
    insist(report["COMPLETED"].toBool(), "Fusion not complete");
    insist(report["EDGES"].toArray().size() == 12, "Neighbour graph lacks 3D loops");
    auto positions = report["POSITIONS"].toArray();
    for (int i = 0; i < tiles.size(); ++i)
    {
        auto got = positions[i].toArray(), want = tiles[i].toObject()["VOXEL_ORIGIN"].toArray();
        for (int j = 0; j < 3; ++j)
            insist(std::abs(got[j].toDouble() - want[j].toDouble()) < .4, "Global positions differ from ground truth");
    }
    auto saved = ct::readImageFile(QDir(output).filePath("Slice_0000021.tif"));
    insist(saved.type() == CV_32F, "Float output lost");
    double err = 0;
    int n = 0;
    auto origin = report["OUTPUT_ORIGIN"].toArray();
    for (int y = 0; y < saved.rows; ++y)
        for (int x = 0; x < saved.cols; ++x)
            if (saved.at<float>(y, x) != 0)
            {
                err += std::pow(saved.at<float>(y, x) - phantom(x + origin[0].toDouble(), y + origin[1].toDouble(),
                                                                20 + origin[2].toDouble()),
                                2);
                ++n;
            }
    double rmse = std::sqrt(err / std::max(1, n));
    std::cout << "Synthetic fusion RMSE=" << rmse << std::endl;
    insist(n > 100 && rmse < .08, "Fusion intensity or placement regression");
    QString repeat = QDir(root).filePath("RepeatRecon");
    ct::stitchMosaic(input, repeat, "Recon", QDir(output).filePath("reconstruction.json"), stop, log);
    insist(read(QDir(repeat).filePath("reconstruction.json"))["POSITIONS"] == report["POSITIONS"],
           "Frozen geometry changed");
    // Fail closed for incomplete data and mixed reconstruction settings.
    auto rejects = [&](const QString &out) {
        bool rejected = false;
        try
        {
            ct::stitchMosaic(input, QDir(root).filePath(out), "Recon", {}, stop, log);
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        insist(rejected, "Invalid input was fused");
        insist(!QFileInfo::exists(QDir(root).filePath(out)), "Invalid input created output");
    };
    QString firstDir = QDir(root).filePath("Tile_0001/Recon");
    auto settings = read(QDir(firstDir).filePath("recon_params.json"));
    auto mixed = settings;
    mixed["PHASE"] = QJsonObject{{"ENABLE", true}, {"ALPHA_RATIO", 123}};
    json(QDir(firstDir).filePath("recon_params.json"), mixed);
    rejects("MixedSettings");
    json(QDir(firstDir).filePath("recon_params.json"), settings);
    QString missing = QDir(firstDir).filePath("Slice_0007.tiff");
    insist(QFile::rename(missing, missing + ".held"), "Missing slice fixture failed");
    rejects("MissingSlice");
    insist(QFile::rename(missing + ".held", missing), "Slice restore failed");
    auto noisy = volume(64, 32, truth, 1.15, .2);
    cv::RNG rng(20261116);
    for (auto &s : noisy)
    {
        cv::Mat noise(s.size(), CV_32F);
        rng.fill(noise, cv::RNG::NORMAL, 0, .015);
        s += noise;
    }
    auto noisyEdge = ct::registerMosaicPair(a, noisy, {24, 0, 0}, 3, .8);
    insist(cv::norm(noisyEdge.delta - truth) < .45, "Noisy subvoxel registration failed");
    // Coarse previews plus full-resolution refinement must retain voxel accuracy.
    auto coarse = manifest;
    auto coarseConfig = c;
    coarseConfig.previewMax = 16;
    coarse["CONFIG"] = coarseConfig.toJson();
    QString coarseManifest = QDir(root).filePath("coarse_manifest.json");
    json(coarseManifest, coarse);
    QString coarseOut = QDir(root).filePath("CoarseRecon");
    ct::stitchMosaic(coarseManifest, coarseOut, "Recon", {}, stop, log);
    const auto coarsePositions = read(QDir(coarseOut).filePath("reconstruction.json"))["POSITIONS"].toArray();
    for (int i = 0; i < tiles.size(); ++i)
        for (int j = 0; j < 3; ++j)
            insist(std::abs(coarsePositions[i].toArray()[j].toDouble() - positions[i].toArray()[j].toDouble()) < .5,
                   "Coarse-to-full resolution refinement drifted");
    // Stop during output leaves an explicit incomplete report, never a done result.
    QString stoppedOut = QDir(root).filePath("StoppedRecon");
    bool cancelled = false;
    try
    {
        ct::stitchMosaic(input, stoppedOut, "Recon", {}, stop, [&](const QString &s) {
            if (s.startsWith("Mosaic saved 1/"))
                stop = true;
        });
    }
    catch (const std::exception &)
    {
        cancelled = true;
    }
    insist(cancelled && !read(QDir(stoppedOut).filePath("reconstruction.json"))["COMPLETED"].toBool(),
           "Stopped fusion was marked complete");
    stop = false;
    // Exercise the real planner's negative motor coordinates and physical crop.
    ct::MosaicConfig plannedConfig = c;
    plannedConfig.extentX0Um = 120;
    plannedConfig.extentX90Um = 20;
    plannedConfig.extentZUm = 40;
    auto planned = ct::makeMosaicPlan(plannedConfig, 64, 32);
    QString plannedRoot = QDir(root).filePath("Planned");
    QDir().mkpath(plannedRoot);
    for (const auto &tile : planned.tiles)
    {
        const QString dir = QDir(plannedRoot).filePath(tile.directory + "/Recon");
        QDir().mkpath(dir);
        auto data = volume(64, 32, tile.voxelOrigin);
        for (int z = 0; z < data.size(); ++z)
        {
            QString error;
            insist(ct::writeImageFile(QDir(dir).filePath(QString("Slice_%1.tiff").arg(z, 4, 10, QLatin1Char('0'))),
                                      data[z], &error),
                   "Planned TIFF save failed");
        }
        json(QDir(dir).filePath("recon_result.json"), {{"STATUS", "done"}});
        json(QDir(dir).filePath("recon_params.json"), settings);
    }
    auto plannedManifest = planned.toJson();
    plannedManifest["COMPLETED"] = true;
    QString plannedInput = QDir(plannedRoot).filePath("mosaic_manifest.json");
    json(plannedInput, plannedManifest);
    QString plannedOutput = QDir(plannedRoot).filePath("MosaicRecon");
    ct::stitchMosaic(plannedInput, plannedOutput, "Recon", {}, stop, log);
    auto pr = read(QDir(plannedOutput).filePath("reconstruction.json"));
    auto slice = ct::readImageFile(QDir(plannedOutput).filePath("Slice_0000021.tif"));
    auto po = pr["OUTPUT_ORIGIN"].toArray();
    insist(slice.cols <= 123 && slice.rows <= 23, "Physical target crop ignored");
    for (int y = 2; y < slice.rows - 2; ++y)
        for (int x = 2; x < slice.cols - 2; ++x)
            insist(std::abs(slice.at<float>(y, x) -
                            phantom(x + po[0].toDouble(), y + po[1].toDouble(), 20 + po[2].toDouble())) < .13,
                   "Planned coverage has hole or misplaced structure");
    std::cout << "PASS: planner, invalid inputs, subvoxel 3D registration, blank rejection, loop rejection, "
                 "cancellation, 2x2x2 float TIFF fusion, photometry, frozen geometry."
              << std::endl;
}
void makeProjections(QString root)
{
    insist(!QFileInfo::exists(root), "Projection fixture directory must be new");
    insist(QDir().mkpath(root), "Cannot create projection fixture");
    constexpr int w = 64, h = 32, angles = 180;
    ct::MosaicConfig mc;
    mc.calibrated = true;
    mc.pixelUm = 1;
    mc.usableDiameter = .9;
    mc.previewMax = 64;
    mc.minNcc = .75;
    mc.searchRadius = 3;
    ct::AppConfig cfg;
    cfg.geometry.volW = w;
    cfg.geometry.volH = h;
    cfg.geometry.totalAngles = angles;
    cfg.io.reconMethod = ct::ReconMethod::OfflineFBP;
    cfg.io.correctionMode = ct::CorrectionMode::FlatDark;
    cfg.io.saveDepth = ct::SaveDepth::Bit32;
    cfg.io.saveFormat = "tiff";
    cfg.io.filePrefix = "Slice";
    cfg.compute.batchSize = 32;
    cfg.compute.zChunkSize = 8;
    cfg.io.tiffThreads = 2;
    cfg.phase.enable = false;
    cfg.phase.pixelSizeUm = 1;
    cfg.preprocess.flatGap = 0;
    cfg.preprocess.flatNum = 1;
    cfg.preprocess.taperWidth = 0;
    cfg.mosaic = mc;
    json(QDir(root).filePath("test_config.json"), cfg.toJsonObject());
    QJsonArray tiles;
    // Analytic line integrals through asymmetric 3-D Gaussian structures.
    struct Blob
    {
        double x, y, z, s, sz, mu;
    };
    const Blob blobs[]{{-7, -5, 10, 3, 5, .018},  {8, 6, 20, 2.5, 4, .025},  {1, -9, 16, 2, 3, .03},
                       {-3, 10, 23, 3.5, 5, .02}, {10, -2, 7, 2.2, 4, .024}, {-10, 3, 17, 2, 3, .02}};
    for (int tile = 0; tile < 2; ++tile)
    {
        ct::MosaicTile t;
        t.id = tile + 1;
        t.ix = tile;
        t.directory = QString("Tile_%1").arg(t.id, 4, 10, QLatin1Char('0'));
        // OfflineFBP uses detector u = x*sin(theta) - y*cos(theta).
        // The analytic physical X translation therefore maps to recon -Y.
        t.voxelOrigin = {0, -tile * 12., 0};
        tiles.append(t.toJson());
        QString dir = QDir(root).filePath(t.directory);
        QDir().mkpath(dir);
        for (int a = 0; a <= angles; ++a)
        {
            double theta = a * std::acos(-1.) / angles;
            cv::Mat im(h, w, CV_16U);
            for (int z = 0; z < h; ++z)
                for (int u = 0; u < w; ++u)
                {
                    double integral = 0;
                    for (auto b : blobs)
                    {
                        double projected = (b.x - tile * 12.) * std::cos(theta) + b.y * std::sin(theta);
                        double d = u - (w - 1) / 2. - projected;
                        integral += b.mu * std::sqrt(2 * std::acos(-1.)) * b.s *
                                    std::exp(-d * d / (2 * b.s * b.s) - (z - b.z) * (z - b.z) / (2 * b.sz * b.sz));
                    }
                    im.at<uint16_t>(z, u) = uint16_t(std::round(100 + 50000 * std::exp(-integral)));
                }
            QString file =
                a == angles ? "alignment180.tif" : QString("Tomo_%1.tif").arg(a + 1, 7, 10, QLatin1Char('0'));
            QString err;
            insist(ct::writeImageFile(QDir(dir).filePath(file), im, &err), "Projection save failed");
        }
        QString err;
        insist(
            ct::writeImageFile(QDir(dir).filePath("Flat_0000001.tif"), cv::Mat(h, w, CV_16U, cv::Scalar(50100)), &err),
            "Flat save failed");
        insist(ct::writeImageFile(QDir(dir).filePath("Dark_0000001.tif"), cv::Mat(h, w, CV_16U, cv::Scalar(100)), &err),
               "Dark save failed");
        json(QDir(dir).filePath("scan_manifest.json"), {{"COMPLETED", true},
                                                        {"ACQUISITION", QJsonObject{{"IMAGE_WIDTH", w},
                                                                                    {"IMAGE_HEIGHT", h},
                                                                                    {"PROJECTION_COUNT", angles},
                                                                                    {"BEGINNING_ANGLE_DEG", 0},
                                                                                    {"FLAT_GAP", 0},
                                                                                    {"FLAT_NUM", 1}}}});
    }
    json(QDir(root).filePath("mosaic_manifest.json"), {{"TYPE", "CT_STREAM_MOSAIC"},
                                                       {"VERSION", 1},
                                                       {"COMPLETED", true},
                                                       {"IMAGE_WIDTH", w},
                                                       {"IMAGE_HEIGHT", h},
                                                       {"CONFIG", mc.toJson()},
                                                       {"TILES", tiles}});
    std::cout << "Synthetic raw projection fixture created: 2 tiles, 180 angles, flats and darks\n";
}
} // namespace
int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    auto args = app.arguments();
    try
    {
        if (args.size() == 4 && args[1] == "--check-slices")
        {
            const QStringList files = QDir(args[2]).entryList({"Slice_*.tif", "Slice_*.tiff"}, QDir::Files, QDir::Name);
            insist(!files.empty(), "No slices to verify");
            const int expectedType = args[3] == "32" ? CV_32F : CV_16U;
            double low = 1e300, high = -1e300;
            cv::Size size;
            for (const auto &name : files)
            {
                auto s = ct::readImageFile(QDir(args[2]).filePath(name));
                insist(s.type() == expectedType && cv::checkRange(s), "Slice type/finite check failed");
                if (size.empty())
                    size = s.size();
                insist(size == s.size(), "Slice dimensions differ");
                double a, b;
                cv::minMaxLoc(s, &a, &b);
                low = std::min(low, a);
                high = std::max(high, b);
            }
            insist(high > low, "Empty/constant reconstruction");
            std::cout << "Verified " << files.size() << " slices, " << size.width << "x" << size.height
                      << ", type=" << args[3].toStdString() << ", min=" << low << ", max=" << high << '\n';
            return 0;
        }
        if (args.size() >= 3 && args[1] == "--self-test")
        {
            selfTest(args[2]);
            return 0;
        }
        if (args.size() == 3 && args[1] == "--make-projections")
        {
            makeProjections(args[2]);
            return 0;
        }
        if (args.size() >= 5 && args[1] == "--stitch")
        {
            auto stop = std::make_shared<std::atomic_bool>(false);
            std::thread([stop]() {
                std::string line;
                while (std::getline(std::cin, line))
                {
                    if (line == "stop")
                    {
                        stop->store(true);
                        break;
                    }
                }
            }).detach();
            ct::stitchMosaic(args[2], args[3], args[4], args.size() > 5 ? args[5] : QString(), *stop,
                             [](const QString &s) { std::cout << s.toStdString() << std::endl; });
            return 0;
        }
        if (args.size() == 2 && args[1] == "--test")
        {
            QTemporaryDir temp;
            selfTest(temp.filePath("data"));
            return 0;
        }
        std::cerr << "Usage: ct_mosaic --stitch manifest.json NEW_OUTPUT_DIR TILE_RECON_NAME [registration.json]\n     "
                     "  ct_mosaic --self-test NEW_TEST_DIR\n";
        return 2;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Mosaic failed: " << e.what() << std::endl;
        return 1;
    }
}
