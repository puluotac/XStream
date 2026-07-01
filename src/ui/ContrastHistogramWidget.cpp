#include "ui/ContrastHistogramWidget.h"

#include <QMouseEvent>
#include <QPainter>

#include <algorithm>
#include <cmath>

namespace ct
{

namespace
{

constexpr int kBinCount = 50;
constexpr int kMaxHistogramSamples = 10000;

} // namespace

ContrastHistogramWidget::ContrastHistogramWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(150);
    setMouseTracking(true);
}

void ContrastHistogramWidget::resetWindow()
{
    bins_.assign(kBinCount, 0.0);
    dataMin_ = -10.0;
    dataMax_ = 10.0;
    vmin_ = -10.0;
    vmax_ = 10.0;
    hasWindow_ = true;
    initializedFromData_ = false;
    dragging_ = DragTarget::None;
    update();
}

void ContrastHistogramWidget::updateData(const cv::Mat& image)
{
    rebuildHistogram(image);

    if (!initializedFromData_ && dataMax_ > dataMin_)
    {
        vmin_ = dataMin_;
        vmax_ = dataMax_;
        hasWindow_ = true;
        initializedFromData_ = true;
        emit windowChanged(vmin_, vmax_);
    }
    else if (initializedFromData_)
    {
        const double oldMin = vmin_;
        const double oldMax = vmax_;
        sanitizeWindowForData();
        if (vmin_ != oldMin || vmax_ != oldMax)
        {
            emit windowChanged(vmin_, vmax_);
        }
    }

    update();
}

void ContrastHistogramWidget::setWindow(const PreviewWindow& window, bool notify)
{
    if (!window.isValid())
    {
        return;
    }
    vmin_ = window.lower;
    vmax_ = window.upper;
    sanitizeWindowForData();
    hasWindow_ = true;
    initializedFromData_ = true;
    if (notify)
    {
        emit windowChanged(vmin_, vmax_);
    }
    update();
}

PreviewWindow ContrastHistogramWidget::window() const
{
    return { vmin_, vmax_ };
}

bool ContrastHistogramWidget::hasWindow() const
{
    return hasWindow_;
}

bool ContrastHistogramWidget::isInitializedFromData() const
{
    return initializedFromData_;
}

void ContrastHistogramWidget::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor("#1e1e1e"));
    painter.setRenderHint(QPainter::Antialiasing, false);

    const QRectF plot = plotRect();
    painter.fillRect(plot, QColor("#171717"));
    painter.setPen(QColor("#444444"));
    painter.drawRect(plot);

    const double maxBin = bins_.empty() ? 0.0 : *std::max_element(bins_.begin(), bins_.end());
    if (maxBin > 0.0)
    {
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 122, 204, 155));
        const double binWidth = plot.width() / static_cast<double>(bins_.size());
        for (size_t i = 0; i < bins_.size(); ++i)
        {
            const double h = (bins_[i] / maxBin) * plot.height();
            const QRectF bar(plot.left() + static_cast<double>(i) * binWidth,
                plot.bottom() - h,
                std::max(1.0, binWidth - 1.0),
                h);
            painter.drawRect(bar);
        }
    }

    auto drawMarker = [&](double value, const QColor& color, const QString& text) {
        const double x = valueToX(value);
        painter.setPen(QPen(color, 2, Qt::DashLine));
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
        painter.setPen(color);
        painter.drawText(QPointF(x + 4.0, plot.top() + 14.0), text);
    };

    drawMarker(vmin_, QColor("#ffcc00"), QStringLiteral("Min"));
    drawMarker(vmax_, QColor("#ff3333"), QStringLiteral("Max"));

    painter.setPen(QColor("#888888"));
    painter.drawText(QRectF(8, height() - 22, width() - 16, 18),
        Qt::AlignLeft | Qt::AlignVCenter,
        QStringLiteral("Data: %1 .. %2    Window: %3 .. %4")
            .arg(dataMin_, 0, 'g', 5)
            .arg(dataMax_, 0, 'g', 5)
            .arg(vmin_, 0, 'g', 5)
            .arg(vmax_, 0, 'g', 5));
}

void ContrastHistogramWidget::mousePressEvent(QMouseEvent* event)
{
    const QRectF plot = plotRect();
    if (!plot.contains(event->position()))
    {
        return;
    }

    const double x = event->position().x();
    const double distMin = std::abs(x - valueToX(vmin_));
    const double distMax = std::abs(x - valueToX(vmax_));
    const double tolerance = plot.width() * 0.05;

    if (distMin < tolerance && distMin < distMax)
    {
        dragging_ = DragTarget::Min;
    }
    else if (distMax < tolerance)
    {
        dragging_ = DragTarget::Max;
    }
}

void ContrastHistogramWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (dragging_ == DragTarget::None)
    {
        return;
    }

    const double value = xToValue(event->position().x());
    if (dragging_ == DragTarget::Min && value < vmax_)
    {
        vmin_ = std::clamp(value, dataMin_, dataMax_);
    }
    else if (dragging_ == DragTarget::Max && value > vmin_)
    {
        vmax_ = std::clamp(value, dataMin_, dataMax_);
    }
    sanitizeWindowForData();

    hasWindow_ = true;
    initializedFromData_ = true;
    emit windowChanged(vmin_, vmax_);
    update();
}

void ContrastHistogramWidget::mouseReleaseEvent(QMouseEvent*)
{
    dragging_ = DragTarget::None;
}

QRectF ContrastHistogramWidget::plotRect() const
{
    return QRectF(8.0, 8.0, std::max(1, width() - 16), std::max(1, height() - 38));
}

double ContrastHistogramWidget::valueToX(double value) const
{
    const QRectF plot = plotRect();
    const double range = (dataMax_ > dataMin_) ? (dataMax_ - dataMin_) : 1.0;
    const double t = (value - dataMin_) / range;
    return plot.left() + std::clamp(t, 0.0, 1.0) * plot.width();
}

double ContrastHistogramWidget::xToValue(double x) const
{
    const QRectF plot = plotRect();
    const double t = (x - plot.left()) / std::max(1.0, plot.width());
    return dataMin_ + std::clamp(t, 0.0, 1.0) * ((dataMax_ > dataMin_) ? (dataMax_ - dataMin_) : 1.0);
}

void ContrastHistogramWidget::sanitizeWindowForData()
{
    if (!std::isfinite(dataMin_) || !std::isfinite(dataMax_) || dataMax_ <= dataMin_)
    {
        return;
    }
    if (!std::isfinite(vmin_) || !std::isfinite(vmax_) || vmax_ <= vmin_)
    {
        vmin_ = dataMin_;
        vmax_ = dataMax_;
        return;
    }

    const double dataRange = std::max(1e-9, dataMax_ - dataMin_);
    const double windowRange = std::clamp(vmax_ - vmin_, dataRange * 1e-6, dataRange);
    if (vmax_ <= dataMin_ || vmin_ >= dataMax_)
    {
        vmin_ = dataMin_;
        vmax_ = dataMax_;
        return;
    }

    vmin_ = std::clamp(vmin_, dataMin_, dataMax_);
    vmax_ = std::clamp(vmax_, dataMin_, dataMax_);
    if (vmax_ <= vmin_)
    {
        const double center = std::clamp((vmin_ + vmax_) * 0.5, dataMin_, dataMax_);
        vmin_ = std::clamp(center - windowRange * 0.5, dataMin_, dataMax_);
        vmax_ = std::clamp(vmin_ + windowRange, dataMin_, dataMax_);
        if (vmax_ <= vmin_)
        {
            vmin_ = dataMin_;
            vmax_ = dataMax_;
        }
    }
}

void ContrastHistogramWidget::rebuildHistogram(const cv::Mat& image)
{
    bins_.assign(kBinCount, 0.0);
    if (image.empty())
    {
        return;
    }

    const int total = image.rows * image.cols;
    const int stride = std::max(1, total / kMaxHistogramSamples);
    std::vector<float> samples;
    samples.reserve(std::min(total, kMaxHistogramSamples));

    int index = 0;
    for (int y = 0; y < image.rows; ++y)
    {
        const float* row = image.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x, ++index)
        {
            if ((index % stride) != 0)
            {
                continue;
            }
            const float value = row[x];
            if (std::isfinite(value))
            {
                samples.push_back(value);
            }
        }
    }

    if (samples.empty())
    {
        dataMin_ = -10.0;
        dataMax_ = 10.0;
        return;
    }

    const auto [minIt, maxIt] = std::minmax_element(samples.begin(), samples.end());
    dataMin_ = *minIt;
    dataMax_ = *maxIt;
    if (dataMax_ <= dataMin_)
    {
        dataMax_ = dataMin_ + 1.0;
    }

    const double range = dataMax_ - dataMin_;
    for (const float value : samples)
    {
        const double normalized = (static_cast<double>(value) - dataMin_) / range;
        const int bin = std::clamp(static_cast<int>(normalized * kBinCount), 0, kBinCount - 1);
        bins_[static_cast<size_t>(bin)] += 1.0;
    }
}

} // namespace ct
