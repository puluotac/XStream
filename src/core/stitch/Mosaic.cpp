#include "core/stitch/Mosaic.h"
#include "core/io/ImageFileIo.h"
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QSet>
#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace ct
{
namespace
{
void require(bool yes, const QString &message)
{
    if (!yes)
        throw std::runtime_error(message.toStdString());
}
void checkStop(const std::atomic_bool *stop)
{
    require(!stop || !stop->load(), "Mosaic stopped by user.");
}
QJsonArray vec(cv::Vec3d v)
{
    return {v[0], v[1], v[2]};
}
cv::Vec3d unvec(QJsonArray a)
{
    require(a.size() == 3, "Invalid mosaic coordinate.");
    for (const auto &v : a)
        require(v.isDouble() && std::isfinite(v.toDouble()), "Non-finite mosaic coordinate.");
    return {a[0].toDouble(), a[1].toDouble(), a[2].toDouble()};
}
QJsonObject readJson(QString p)
{
    QFile f(p);
    require(f.open(QIODevice::ReadOnly), "Cannot read " + p);
    QJsonParseError e;
    auto d = QJsonDocument::fromJson(f.readAll(), &e);
    require(e.error == QJsonParseError::NoError && d.isObject(), "Invalid JSON " + p);
    return d.object();
}
void writeJson(QString p, QJsonObject o)
{
    QSaveFile f(p);
    require(f.open(QIODevice::WriteOnly), "Cannot write " + p);
    auto b = QJsonDocument(o).toJson();
    require(f.write(b) == b.size() && f.commit(), "Cannot commit " + p);
}
QVector<double> centres(double extent, double core, double overlap)
{
    double count = extent <= core ? 1 : 1 + std::ceil((extent - core) / (core * (1 - overlap)));
    require(std::isfinite(count) && count <= 1000, "Mosaic axis requires more than 1000 positions.");
    int n = int(count);
    QVector<double> r;
    for (int i = 0; i < n; ++i)
        r.append(n == 1 ? 0 : (i - double(n - 1) / 2) * (extent - core) / (n - 1));
    return r;
}
struct Moments
{
    double n = 0, a = 0, b = 0, aa = 0, bb = 0, ab = 0;
    void add(double x, double y)
    {
        if (std::isfinite(x) && std::isfinite(y))
        {
            ++n;
            a += x;
            b += y;
            aa += x * x;
            bb += y * y;
            ab += x * y;
        }
    }
    double score() const
    {
        double va = aa - a * a / std::max(1., n), vb = bb - b * b / std::max(1., n);
        return n >= 64 && va > 1e-12 && vb > 1e-12 ? (ab - a * b / n) / std::sqrt(va * vb) : -2;
    }
};
float sample(const QVector<cv::Mat> &v, double x, double y, double z)
{
    int ix = int(std::floor(x)), iy = int(std::floor(y)), iz = int(std::floor(z));
    if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= v[0].cols || iy + 1 >= v[0].rows || iz + 1 >= v.size())
        return std::numeric_limits<float>::quiet_NaN();
    double out = 0;
    for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i)
                out += v[iz + k].at<float>(iy + j, ix + i) * (i ? x - ix : 1 - x + ix) * (j ? y - iy : 1 - y + iy) *
                       (k ? z - iz : 1 - z + iz);
    return float(out);
}
Moments compare(const QVector<cv::Mat> &a, const QVector<cv::Mat> &b, cv::Vec3d d, int stride)
{
    Moments m;
    int x0 = std::max(1, int(std::ceil(d[0])) + 1), y0 = std::max(1, int(std::ceil(d[1])) + 1),
        z0 = std::max(1, int(std::ceil(d[2])) + 1);
    int x1 = std::min(a[0].cols - 1, int(std::floor(d[0] + b[0].cols - 1))),
        y1 = std::min(a[0].rows - 1, int(std::floor(d[1] + b[0].rows - 1))),
        z1 = std::min(int(a.size()) - 1, int(std::floor(d[2] + b.size() - 1)));
    for (int z = z0; z < z1; z += stride)
        for (int y = y0; y < y1; y += stride)
            for (int x = x0; x < x1; x += stride)
                m.add(a[z].at<float>(y, x), sample(b, x - d[0], y - d[1], z - d[2]));
    return m;
}
} // namespace
QJsonObject MosaicConfig::toJson() const
{
    return {{"PIXEL_UM", pixelUm},
            {"EXTENT_X0_UM", extentX0Um},
            {"EXTENT_X90_UM", extentX90Um},
            {"EXTENT_Z_UM", extentZUm},
            {"OVERLAP", overlap},
            {"USABLE_DIAMETER", usableDiameter},
            {"SETTLE_MS", settleMs},
            {"FLAT_MOVE_UM", flatMoveUm},
            {"FLAT_COUNT", flatCount},
            {"DARK_PATH", darkPath},
            {"CALIBRATED", calibrated},
            {"MOTOR_TO_RECON_DEG", motorToReconDeg},
            {"X90_SIGN", x90Sign},
            {"Z_SIGN", zSign},
            {"PREVIEW_MAX", previewMax},
            {"SEARCH_RADIUS", searchRadius},
            {"MIN_NCC", minNcc},
            {"MAX_RESIDUAL_PX", maxResidualPx}};
}
MosaicConfig MosaicConfig::fromJson(const QJsonObject &o)
{
    MosaicConfig c;
#define D(key, field) c.field = o.value(key).toDouble(c.field)
    D("PIXEL_UM", pixelUm);
    D("EXTENT_X0_UM", extentX0Um);
    D("EXTENT_X90_UM", extentX90Um);
    D("EXTENT_Z_UM", extentZUm);
    D("OVERLAP", overlap);
    D("USABLE_DIAMETER", usableDiameter);
    D("MOTOR_TO_RECON_DEG", motorToReconDeg);
    D("MIN_NCC", minNcc);
    D("MAX_RESIDUAL_PX", maxResidualPx);
    D("FLAT_MOVE_UM", flatMoveUm);
#undef D
    c.settleMs = o.value("SETTLE_MS").toInt(c.settleMs);
    c.flatCount = o.value("FLAT_COUNT").toInt(c.flatCount);
    c.darkPath = o.value("DARK_PATH").toString();
    c.calibrated = o.value("CALIBRATED").toBool(false);
    c.x90Sign = o.value("X90_SIGN").toInt(c.x90Sign);
    c.zSign = o.value("Z_SIGN").toInt(c.zSign);
    c.previewMax = o.value("PREVIEW_MAX").toInt(c.previewMax);
    c.searchRadius = o.value("SEARCH_RADIUS").toInt(c.searchRadius);
    return c;
}
QJsonObject MosaicTile::toJson() const
{
    return {{"ID", id},
            {"IX", ix},
            {"IY", iy},
            {"IZ", iz},
            {"MOTOR_OFFSET_UM", vec(motorOffset)},
            {"VOXEL_ORIGIN", vec(voxelOrigin)},
            {"DIRECTORY", directory}};
}
QJsonObject MosaicPlan::toJson() const
{
    QJsonArray a;
    for (auto &t : tiles)
        a.append(t.toJson());
    return {{"TYPE", "CT_STREAM_MOSAIC"}, {"VERSION", 1},         {"CONFIG", config.toJson()}, {"IMAGE_WIDTH", width},
            {"IMAGE_HEIGHT", height},     {"CORE_X_UM", coreXUm}, {"CORE_Y_UM", coreYUm},      {"TILES", a}};
}
MosaicPlan makeMosaicPlan(const MosaicConfig &c, int w, int h)
{
    require(w >= 8 && h >= 4 && std::isfinite(c.pixelUm) && c.pixelUm > 0,
            "Invalid mosaic pixel size or image dimensions.");
    require(std::isfinite(c.extentX0Um) && std::isfinite(c.extentX90Um) && std::isfinite(c.extentZUm) &&
                c.extentX0Um > 0 && c.extentX90Um > 0 && c.extentZUm > 0,
            "Sample extents must be finite and positive.");
    require(c.overlap >= 0.1 && c.overlap <= 0.8 && c.usableDiameter >= 0.5 && c.usableDiameter <= 0.98,
            "Invalid overlap or usable diameter.");
    require(std::isfinite(c.motorToReconDeg) && std::abs(c.x90Sign) == 1 && std::abs(c.zSign) == 1 && c.settleMs >= 0 &&
                c.settleMs <= 60000,
            "Invalid axis calibration or settling time.");
    MosaicPlan p;
    p.config = c;
    p.width = w;
    p.height = h;
    const double diameter = w * c.pixelUm * c.usableDiameter;
    p.coreYUm = std::min(c.extentX90Um, diameter / std::sqrt(2.));
    p.coreXUm = std::sqrt(diameter * diameter - p.coreYUm * p.coreYUm);
    auto xs = centres(c.extentX0Um, p.coreXUm, c.overlap), ys = centres(c.extentX90Um, p.coreYUm, c.overlap),
         zs = centres(c.extentZUm, h * c.pixelUm * c.usableDiameter, c.overlap);
    require(qint64(xs.size()) * ys.size() * zs.size() <= 10000, "Plan exceeds 10000 tiles.");
    double ang = c.motorToReconDeg * std::acos(-1.) / 180, cs = std::cos(ang), sn = std::sin(ang);
    for (int z = 0; z < zs.size(); ++z)
        for (int y = 0; y < ys.size(); ++y)
            for (int k = 0; k < xs.size(); ++k)
            {
                int x = ((z * ys.size() + y) % 2) ? xs.size() - 1 - k : k;
                MosaicTile t;
                t.id = p.tiles.size() + 1;
                t.ix = x;
                t.iy = y;
                t.iz = z;
                t.motorOffset = {xs[x], ys[y], zs[z]};
                // Moving the specimen by +t exposes a region centred at -t in specimen coordinates.
                t.voxelOrigin = {-(cs * xs[x] - sn * c.x90Sign * ys[y]) / c.pixelUm,
                                 -(sn * xs[x] + cs * c.x90Sign * ys[y]) / c.pixelUm, -c.zSign * zs[z] / c.pixelUm};
                t.directory = QString("Tile_%1").arg(t.id, 4, 10, QLatin1Char('0'));
                p.tiles.append(t);
            }
    return p;
}
MosaicEdge registerMosaicPair(const QVector<cv::Mat> &a, const QVector<cv::Mat> &b, cv::Vec3d nominal, int radius,
                              double minNcc, const std::atomic_bool *stop)
{
    require(a.size() >= 4 && b.size() >= 4 && radius >= 0 && radius <= 32,
            "Invalid registration volume or search radius.");
    for (auto *v : {&a, &b})
        for (auto &s : *v)
            require(s.type() == CV_32F && s.size() == v->first().size(),
                    "Registration requires consistent float32 slices.");
    cv::Vec3d best = nominal;
    double score = -2;
    // Phase correlation on a mean overlap projection supplies an independent
    // XY seed. Score both signs against the 3-D objective; never trust a Fourier
    // peak on a repetitive or empty specimen without spatial verification.
    int ax = std::max(0, int(std::round(nominal[0]))), bx = std::max(0, -int(std::round(nominal[0])));
    int ay = std::max(0, int(std::round(nominal[1]))), by = std::max(0, -int(std::round(nominal[1])));
    int az = std::max(0, int(std::round(nominal[2]))), bz = std::max(0, -int(std::round(nominal[2])));
    int pw = std::min(a[0].cols - ax, b[0].cols - bx), ph = std::min(a[0].rows - ay, b[0].rows - by),
        pz = std::min(int(a.size()) - az, int(b.size()) - bz);
    if (pw >= 16 && ph >= 16 && pz >= 4)
    {
        cv::Mat pa = cv::Mat::zeros(ph, pw, CV_32F), pb = pa.clone(), window;
        for (int z = 0; z < pz; ++z)
        {
            pa += a[az + z](cv::Rect(ax, ay, pw, ph));
            pb += b[bz + z](cv::Rect(bx, by, pw, ph));
        }
        pa /= pz;
        pb /= pz;
        cv::patchNaNs(pa, 0);
        cv::patchNaNs(pb, 0);
        pa -= cv::mean(pa)[0];
        pb -= cv::mean(pb)[0];
        cv::createHanningWindow(window, pa.size(), CV_32F);
        double response = 0;
        auto shift = cv::phaseCorrelate(pa, pb, window, &response);
        if (response > 0.1 && std::isfinite(shift.x) && std::isfinite(shift.y) && std::abs(shift.x) <= radius &&
            std::abs(shift.y) <= radius)
        {
            for (int sign : {-1, 1})
            {
                auto d = nominal + cv::Vec3d(sign * shift.x, sign * shift.y, 0);
                auto value = compare(a, b, d, 3).score();
                if (value > score)
                {
                    score = value;
                    best = d;
                }
            }
        }
    }

    // Bounded 3-D coarse NCC search: motor geometry limits correspondence ambiguity.
    for (int z = -radius; z <= radius; ++z)
    {
        checkStop(stop);
        for (int y = -radius; y <= radius; ++y)
            for (int x = -radius; x <= radius; ++x)
            {
                cv::Vec3d d = nominal + cv::Vec3d(x, y, z);
                double n = compare(a, b, d, 3).score();
                if (n > score)
                {
                    score = n;
                    best = d;
                }
            }
    }
    // Fine 3-D translation search, with subvoxel trilinear sampling and NCC.
    for (double step : {0.5, 0.25, 0.125})
    {
        auto centre = best;
        score = -2;
        checkStop(stop);
        for (int z = -1; z <= 1; ++z)
            for (int y = -1; y <= 1; ++y)
                for (int x = -1; x <= 1; ++x)
                {
                    auto d = centre + step * cv::Vec3d(x, y, z);
                    bool inside = true;
                    for (int j = 0; j < 3; ++j)
                        inside &= std::abs(d[j] - nominal[j]) <= radius + 0.9;
                    if (!inside)
                        continue;
                    double n = compare(a, b, d, 2).score();
                    if (n > score)
                    {
                        score = n;
                        best = d;
                    }
                }
    }
    auto m = compare(a, b, best, 1);
    require(m.score() >= minNcc,
            QString("Registration rejected: NCC %1 below %2 (blank, ambiguous, or inconsistent overlap).")
                .arg(m.score())
                .arg(minNcc));
    MosaicEdge e;
    e.delta = best;
    e.ncc = m.score();
    e.gain = (m.ab - m.a * m.b / m.n) / (m.bb - m.b * m.b / m.n);
    e.bias = (m.a - e.gain * m.b) / m.n;
    require(e.gain > 0.2 && e.gain < 5, "Registration photometric gain outside allowed range.");
    return e;
}
QVector<cv::Vec3d> solveMosaicPositions(const QVector<cv::Vec3d> &nominal, QVector<MosaicEdge> &edges,
                                        double maxResidual)
{
    int n = nominal.size();
    require(n > 0 && n <= 10000, "Invalid graph size.");
    if (n == 1)
        return nominal;
    QVector<bool> seen(n, false);
    seen[0] = true;
    for (int k = 0; k < n; ++k)
        for (auto &e : edges)
        {
            require(e.a >= 0 && e.b >= 0 && e.a < n && e.b < n && e.a != e.b, "Invalid graph edge.");
            if (seen[e.a] || seen[e.b])
                seen[e.a] = seen[e.b] = true;
        }
    require(std::all_of(seen.begin(), seen.end(), [](bool b) { return b; }),
            "Registration graph is disconnected; no mosaic was fused.");
    QVector<cv::Vec3d> p = nominal; // Sparse neighbour relaxation, anchor tile 0.
    for (int iteration = 0; iteration < 10000; ++iteration)
    {
        double change = 0;
        for (int i = 1; i < n; ++i)
        {
            cv::Vec3d sum(0, 0, 0);
            double weights = 0;
            for (auto &e : edges)
            {
                double w = std::max(0.01, e.ncc * e.ncc);
                if (e.a == i)
                {
                    sum += w * (p[e.b] - e.delta);
                    weights += w;
                }
                else if (e.b == i)
                {
                    sum += w * (p[e.a] + e.delta);
                    weights += w;
                }
            }
            auto next = sum / weights;
            change = std::max(change, cv::norm(next - p[i]));
            p[i] = next;
        }
        if (change < 1e-7)
            break;
    }
    for (auto &e : edges)
    {
        e.residual = cv::norm(p[e.b] - p[e.a] - e.delta);
        require(e.residual <= maxResidual,
                QString("Inconsistent overlap graph: residual %1 voxels exceeds %2.").arg(e.residual).arg(maxResidual));
    }
    return p;
}
} // namespace ct

namespace ct
{
namespace
{
struct DiskTile
{
    QString directory;
    QStringList files;
    QVector<cv::Mat> small;
    cv::Vec3d origin;
    int ix = 0, iy = 0, iz = 0;
};
void maskCircle(cv::Mat &s, double radius, double centre)
{
    for (int y = 0; y < s.rows; ++y)
        for (int x = 0; x < s.cols; ++x)
            if (std::hypot(x - centre, y - centre) >= radius)
                s.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
}
QVector<cv::Mat> downsampleTile(const DiskTile &tile, int w, int h, int scale, double diameter, std::atomic_bool &stop)
{
    QVector<cv::Mat> r;
    // Box averaging in Z and XY, avoiding aliasing of thin neuronal features.
    for (int z = 0; z < h; z += scale)
    {
        checkStop(&stop);
        cv::Mat sum;
        int n = 0;
        for (int k = z; k < std::min(h, z + scale); ++k)
        {
            auto s = readImageFile(tile.files[k]);
            require(s.type() == CV_32F && s.cols == w && s.rows == w,
                    "Mosaic requires unwindowed float32 slices of matching dimensions.");
            require(cv::checkRange(s), "Non-finite tile values.");
            cv::Mat d;
            cv::Mat padded;
            cv::copyMakeBorder(s, padded, 0, (scale - w % scale) % scale, 0, (scale - w % scale) % scale,
                               cv::BORDER_REPLICATE);
            cv::resize(padded, d, cv::Size(padded.cols / scale, padded.rows / scale), 0, 0, cv::INTER_AREA);
            if (sum.empty())
                sum = cv::Mat::zeros(d.size(), CV_32F);
            sum += d;
            ++n;
        }
        sum /= n;
        maskCircle(sum, w * diameter / (2 * scale) - 1, (w / double(scale) - 1) / 2);
        r.append(sum);
    }
    return r;
}
} // namespace
void stitchMosaic(const QString &manifest, const QString &output, const QString &tileReconName,
                  const QString &registeredGeometry, std::atomic_bool &stop, const MosaicLog &log)
{
    const auto m = readJson(manifest);
    require(m.value("TYPE").toString() == "CT_STREAM_MOSAIC" && m.value("COMPLETED").toBool(),
            "Completed mosaic acquisition manifest required.");
    require(!tileReconName.isEmpty() && tileReconName != "." && tileReconName != ".." && !tileReconName.contains('/') &&
                !tileReconName.contains('\\'),
            "Tile reconstruction must be a directory name.");
    require(!QFileInfo::exists(output), "Choose a new mosaic output directory; existing output is never overwritten.");
    auto c = MosaicConfig::fromJson(m.value("CONFIG").toObject());
    int w = m.value("IMAGE_WIDTH").toInt(), h = m.value("IMAGE_HEIGHT").toInt();
    require(c.calibrated, "Motor-to-reconstruction mapping has not been calibrated.");
    require(w >= 8 && h >= 4 && c.pixelUm > 0 && c.previewMax >= 16 && c.previewMax <= 256 && c.searchRadius >= 0 &&
                c.searchRadius <= 32 && c.minNcc > 0 && c.minNcc <= 1 && c.maxResidualPx > 0,
            "Invalid mosaic reconstruction parameters.");
    auto tiles = m.value("TILES").toArray();
    require(!tiles.empty() && tiles.size() <= 10000, "Invalid tile list.");
    QString root = QFileInfo(manifest).absolutePath();
    QVector<DiskTile> inputs;
    QJsonObject referenceSettings;
    QSet<QString> gridCells, directories;
    QVector<cv::Vec3d> positions;
    int scale = std::max(1, (std::max(w, h) + c.previewMax - 1) / c.previewMax);
    double previewBytes = double(tiles.size()) * std::ceil(double(w) / scale) * std::ceil(double(w) / scale) *
                          std::ceil(double(h) / scale) * 4;
    require(previewBytes < 2. * 1024 * 1024 * 1024,
            "Registration previews exceed 2 GiB; reduce preview size or split the dataset.");
    for (auto v : tiles)
    {
        checkStop(&stop);
        auto t = v.toObject();
        DiskTile d;
        QString name = t.value("DIRECTORY").toString();
        require(!name.isEmpty() && !name.contains('/') && !name.contains('\\') && name != ".." && name != ".",
                "Invalid tile path.");
        require(!directories.contains(name), "Duplicate tile directory.");
        directories.insert(name);
        d.directory = QDir(root).filePath(name + "/" + tileReconName);
        auto result = readJson(QDir(d.directory).filePath("recon_result.json"));
        require(result.value("STATUS").toString() == "done", "Tile reconstruction is not complete: " + name);
        auto params = readJson(QDir(d.directory).filePath("recon_params.json"));
        auto prep = params.value("PREPROCESS").toObject();
        prep.remove("FLAT_GROUPS");
        prep.remove("FLAT_GAP");
        prep.remove("FLAT_NUM");
        auto geom = params.value("GEOMETRY").toObject();
        geom.remove("COR_SHIFT");
        geom.remove("COR_AUTO");
        auto io = params.value("IO").toObject();
        if (m.contains("CALIBRATED_RECON_METHOD"))
            require(io.value("RECON_METHOD") == m.value("CALIBRATED_RECON_METHOD"),
                    "Axis calibration is for a different reconstruction method.");
        QJsonObject settings{{"PHASE", params.value("PHASE")},
                             {"PREPROCESS", prep},
                             {"GEOMETRY", geom},
                             {"RECON_METHOD", io.value("RECON_METHOD")},
                             {"CORRECTION_MODE", io.value("CORRECTION_MODE")}};
        if (inputs.isEmpty())
            referenceSettings = settings;
        else
            require(settings == referenceSettings,
                    "Tiles use different reconstruction settings (Alpha, geometry or preprocessing).");
        QString prefix = params.value("IO").toObject().value("FILE_PREFIX").toString("Slice");
        QStringList names =
            QDir(d.directory).entryList({prefix + "_*.tif", prefix + "_*.tiff"}, QDir::Files, QDir::Name);
        require(names.size() == h, "Tile slice count mismatch: " + name);
        for (int k = 0; k < h; ++k)
        {
            const QString stem = QString("%1_%2").arg(prefix).arg(k, 4, 10, QLatin1Char('0'));
            QString path = QDir(d.directory).filePath(stem + ".tiff");
            if (!QFileInfo::exists(path))
                path = QDir(d.directory).filePath(stem + ".tif");
            require(QFileInfo::exists(path), "Missing numbered tile slice: " + path);
            d.files.append(path);
        }
        d.ix = t.value("IX").toInt();
        d.iy = t.value("IY").toInt();
        d.iz = t.value("IZ").toInt();
        QString cell = QString("%1/%2/%3").arg(d.ix).arg(d.iy).arg(d.iz);
        require(!gridCells.contains(cell), "Duplicate tile grid cell.");
        gridCells.insert(cell);
        d.origin = unvec(t.value("VOXEL_ORIGIN").toArray());
        positions.append(d.origin);
        inputs.append(d);
    }
    const QString fingerprint = QString::fromLatin1(
        QCryptographicHash::hash(QJsonDocument(m).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex());
    QVector<MosaicEdge> edges;
    QVector<double> gains(inputs.size(), 1.), biases(inputs.size(), 0.);
    QJsonArray edgeReport;
    if (!registeredGeometry.isEmpty())
    {
        auto prior = readJson(registeredGeometry);
        require(prior.value("COMPLETED").toBool() && prior.value("SOURCE_SHA256").toString() == fingerprint,
                "Registration belongs to another dataset or is incomplete.");
        auto p = prior.value("POSITIONS").toArray();
        require(p.size() == positions.size(), "Registration tile count mismatch.");
        for (int i = 0; i < p.size(); ++i)
            positions[i] = unvec(p[i].toArray());
        // Geometry is reused exactly for Alpha comparison; radiometry is re-estimated.
        log("Reusing fixed registration geometry; positions will not change with Alpha.");
    }
    log(QString("Loading registration previews: scale=%1, estimated %2 MiB.")
            .arg(scale)
            .arg(previewBytes / (1024 * 1024), 0, 'f', 1));
    for (int i = 0; i < inputs.size(); ++i)
    {
        inputs[i].small = downsampleTile(inputs[i], w, h, scale, c.usableDiameter, stop);
        log(QString("Registration preview %1/%2").arg(i + 1).arg(inputs.size()));
    }
    // Every geometric neighbour is used, including all available rectangular loops.
    for (int i = 0; i < inputs.size(); ++i)
        for (int j = i + 1; j < inputs.size(); ++j)
        {
            auto &a = inputs[i];
            auto &b = inputs[j];
            if (std::abs(a.ix - b.ix) + std::abs(a.iy - b.iy) + std::abs(a.iz - b.iz) != 1)
                continue;
            checkStop(&stop);
            auto nominal = (positions[j] - positions[i]) / scale;
            MosaicEdge e;
            if (registeredGeometry.isEmpty())
                e = registerMosaicPair(a.small, b.small, nominal, c.searchRadius, c.minNcc, &stop);
            else
            {
                auto mm = compare(a.small, b.small, nominal, 1);
                require(mm.score() >= c.minNcc, "Frozen registration no longer matches tile content.");
                e.delta = nominal;
                e.ncc = mm.score();
                e.gain = (mm.ab - mm.a * mm.b / mm.n) / (mm.bb - mm.b * mm.b / mm.n);
                e.bias = (mm.a - e.gain * mm.b) / mm.n;
                require(std::isfinite(e.gain) && std::isfinite(e.bias) && e.gain > .2 && e.gain < 5,
                        "Frozen registration intensity fit is unstable.");
            }
            e.a = i;
            e.b = j;
            e.delta *= scale;
            // Refine in full resolution using a bounded set of slices near overlap centre.
            // Downsampled subvoxel accuracy alone is NOT full-resolution subvoxel accuracy.
            if (registeredGeometry.isEmpty())
            {
                int az = std::max(0, int(std::ceil(e.delta[2]))), bz = std::max(0, int(std::ceil(-e.delta[2])));
                int count = std::min(h - az, h - bz);
                require(count >= 8, "Insufficient full-resolution Z overlap.");
                int dz = std::min(12, count);
                az += (count - dz) / 2;
                bz += (count - dz) / 2;
                QVector<cv::Mat> av, bv;
                // XY patches around the common region keep full-resolution refinement bounded.
                int ax = std::max(0, int(std::ceil(e.delta[0]))), bx = std::max(0, int(std::ceil(-e.delta[0])));
                int ay = std::max(0, int(std::ceil(e.delta[1]))), by = std::max(0, int(std::ceil(-e.delta[1])));
                int pw = std::min({256, w - ax, w - bx}), ph = std::min({256, w - ay, w - by});
                require(pw >= 16 && ph >= 16, "Insufficient full-resolution XY overlap.");
                int offx = (std::min(w - ax, w - bx) - pw) / 2, offy = (std::min(w - ay, w - by) - ph) / 2;
                ax += offx;
                bx += offx;
                ay += offy;
                by += offy;
                for (int k = 0; k < dz; ++k)
                {
                    auto sa = readImageFile(a.files[az + k]), sb = readImageFile(b.files[bz + k]);
                    maskCircle(sa, w * c.usableDiameter / 2 - 1, (w - 1) / 2.);
                    maskCircle(sb, w * c.usableDiameter / 2 - 1, (w - 1) / 2.);
                    av.append(sa(cv::Rect(ax, ay, pw, ph)).clone());
                    bv.append(sb(cv::Rect(bx, by, pw, ph)).clone());
                }
                cv::Vec3d patchOffset(bx - ax, by - ay, bz - az);
                auto fine = registerMosaicPair(av, bv, e.delta + patchOffset, std::min(4, std::max(1, scale / 4)),
                                               c.minNcc, &stop);
                e.delta = fine.delta - patchOffset;
                e.ncc = fine.ncc;
                e.gain = fine.gain;
                e.bias = fine.bias;
            }
            edges.append(e);
            log(QString("Overlap %1-%2: NCC=%3, shift=(%4,%5,%6) px")
                    .arg(i + 1)
                    .arg(j + 1)
                    .arg(e.ncc, 0, 'f', 5)
                    .arg(e.delta[0])
                    .arg(e.delta[1])
                    .arg(e.delta[2]));
        }
    if (registeredGeometry.isEmpty())
        positions = solveMosaicPositions(positions, edges, c.maxResidualPx);
    // Global radiometric fit anchored at tile 0, never per-slice histogram matching.
    for (int it = 0; it < 500; ++it)
    {
        double change = 0;
        for (int i = 1; i < inputs.size(); ++i)
        {
            double g = 0, b = 0, n = 0;
            for (auto &e : edges)
            {
                if (e.b == i)
                {
                    g += gains[e.a] * e.gain;
                    b += gains[e.a] * e.bias + biases[e.a];
                    ++n;
                }
                else if (e.a == i)
                {
                    g += gains[e.b] / e.gain;
                    b += biases[e.b] - gains[e.b] * e.bias / e.gain;
                    ++n;
                }
            }
            require(n > 0, "Disconnected intensity graph.");
            g /= n;
            b /= n;
            change = std::max(change, std::abs(g - gains[i]) + std::abs(b - biases[i]));
            gains[i] = g;
            biases[i] = b;
        }
        if (change < 1e-8)
            break;
    }
    for (int i = 0; i < inputs.size(); ++i)
    {
        require(std::isfinite(gains[i]) && gains[i] > 0.1 && gains[i] < 10 && std::isfinite(biases[i]),
                "Unstable global intensity fit.");
        inputs[i].small.clear();
    }
    cv::Vec3d lo = positions[0], hi = positions[0] + cv::Vec3d(w, w, h);
    for (auto p : positions)
        for (int j = 0; j < 3; ++j)
        {
            lo[j] = std::min(lo[j], std::floor(p[j]));
            hi[j] = std::max(hi[j], std::ceil(p[j] + (j == 2 ? h : w)));
        }
    if (m.contains("CORE_X_UM"))
    {
        const double ang = c.motorToReconDeg * std::acos(-1.) / 180, cs = std::abs(std::cos(ang)),
                     sn = std::abs(std::sin(ang));
        const cv::Vec3d half((cs * c.extentX0Um + sn * c.extentX90Um) / (2 * c.pixelUm),
                             (sn * c.extentX0Um + cs * c.extentX90Um) / (2 * c.pixelUm), c.extentZUm / (2 * c.pixelUm));
        const cv::Vec3d centre((w - 1) / 2., (w - 1) / 2., (h - 1) / 2.);
        for (int j = 0; j < 3; ++j)
        {
            lo[j] = std::max(lo[j], std::floor(centre[j] - half[j]));
            hi[j] = std::min(hi[j], std::ceil(centre[j] + half[j] + 1));
        }
    }
    int outW = int(std::ceil(hi[0] - lo[0])), outH = int(std::ceil(hi[1] - lo[1])),
        outZ = int(std::ceil(hi[2] - lo[2]));
    require(outW > 0 && outH > 0 && outZ > 0, "Invalid output bounds.");
    require(double(outW) * outH * 16 < 4. * 1024 * 1024 * 1024,
            "Output slice working set exceeds 4 GiB; use spatial output chunks.");
    require(QDir().mkpath(output), "Cannot create mosaic output directory.");
    QJsonObject report{{"TYPE", "CT_STREAM_MOSAIC_RECON"},
                       {"VERSION", 1},
                       {"SOURCE_SHA256", fingerprint},
                       {"COMPLETED", false},
                       {"TILE_RECON", tileReconName},
                       {"PIXEL_UM", c.pixelUm},
                       {"OUTPUT_ORIGIN", vec(lo)},
                       {"WIDTH", outW},
                       {"HEIGHT", outH},
                       {"SLICES", outZ},
                       {"BLEND", "circular-feather"}};
    report["TILE_SETTINGS"] = referenceSettings;
    QJsonArray pp, gg, bb;
    for (int i = 0; i < positions.size(); ++i)
    {
        pp.append(vec(positions[i]));
        gg.append(gains[i]);
        bb.append(biases[i]);
    }
    report["POSITIONS"] = pp;
    report["GAINS"] = gg;
    report["BIASES"] = bb;
    for (auto &e : edges)
        edgeReport.append(
            QJsonObject{{"A", e.a}, {"B", e.b}, {"DELTA", vec(e.delta)}, {"NCC", e.ncc}, {"RESIDUAL_PX", e.residual}});
    report["EDGES"] = edgeReport;
    const QString reportPath = QDir(output).filePath("reconstruction.json");
    writeJson(reportPath, report);
    try
    {
        const double radius = w * c.usableDiameter / 2, centre = (w - 1) / 2.;
        for (int z = 0; z < outZ; ++z)
        {
            checkStop(&stop);
            cv::Mat sum = cv::Mat::zeros(outH, outW, CV_32F), weights = cv::Mat::zeros(outH, outW, CV_32F);
            for (int i = 0; i < inputs.size(); ++i)
            {
                auto p = positions[i] - lo;
                double localZ = z - p[2];
                int iz = int(std::floor(localZ));
                if (iz < 0 || iz >= h)
                    continue;
                auto a = readImageFile(inputs[i].files[iz]);
                auto b = iz + 1 < h ? readImageFile(inputs[i].files[iz + 1]) : a;
                require(a.type() == CV_32F && b.type() == CV_32F, "Float tile unavailable during fusion.");
                double fz = localZ - iz;
                int x0 = std::max(0, int(std::ceil(p[0]))), y0 = std::max(0, int(std::ceil(p[1]))),
                    x1 = std::min(outW, int(std::floor(p[0] + w - 1))),
                    y1 = std::min(outH, int(std::floor(p[1] + w - 1)));
                for (int y = y0; y < y1; ++y)
                {
                    checkStop(&stop);
                    double ly = y - p[1];
                    int iy = int(ly);
                    double fy = ly - iy;
                    for (int x = x0; x < x1; ++x)
                    {
                        double lx = x - p[0];
                        double radial = radius - std::hypot(lx - centre, ly - centre);
                        if (radial <= 0)
                            continue;
                        int ix = int(lx);
                        double fx = lx - ix;
                        double v = 0;
                        for (int k = 0; k < 2; ++k)
                        {
                            const auto &s = k ? b : a;
                            double bil = (1 - fy) * ((1 - fx) * s.at<float>(iy, ix) + fx * s.at<float>(iy, ix + 1)) +
                                         fy * ((1 - fx) * s.at<float>(iy + 1, ix) + fx * s.at<float>(iy + 1, ix + 1));
                            v += (k ? fz : 1 - fz) * bil;
                        }
                        require(std::isfinite(v), "Non-finite reconstructed sample during fusion.");
                        double weight =
                            std::min(1., radial / std::max(1., radius * c.overlap)) *
                            std::min(1., std::min(localZ + 1., h - localZ) / std::max(1., h * c.overlap / 2));
                        sum.at<float>(y, x) += float(weight * (v * gains[i] + biases[i]));
                        weights.at<float>(y, x) += float(weight);
                    }
                }
            }
            for (int y = 0; y < outH; ++y)
                for (int x = 0; x < outW; ++x)
                    if (weights.at<float>(y, x) > 0)
                        sum.at<float>(y, x) /= weights.at<float>(y, x);
            QString error;
            require(writeImageFile(QDir(output).filePath(QString("Slice_%1.tif").arg(z + 1, 7, 10, QLatin1Char('0'))),
                                   sum, &error),
                    error);
            if (z % 32 == 0 || z == outZ - 1)
                log(QString("Mosaic saved %1/%2 slices (%3 x %4).").arg(z + 1).arg(outZ).arg(outW).arg(outH));
        }
        report["COMPLETED"] = true;
        writeJson(reportPath, report);
        log("Mosaic fusion completed.");
    }
    catch (const std::exception &e)
    {
        report["ERROR"] = QString::fromUtf8(e.what());
        writeJson(reportPath, report);
        throw;
    }
}
} // namespace ct
