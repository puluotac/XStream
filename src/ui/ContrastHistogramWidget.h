#pragma once

#include "core/common/PreviewSliceRenderer.h"

#include <QRectF>
#include <QWidget>

#include <opencv2/core.hpp>

#include <vector>

class QMouseEvent;
class QPaintEvent;

namespace ct
{

class ContrastHistogramWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ContrastHistogramWidget(QWidget* parent = nullptr);

    void resetWindow();
    void updateData(const cv::Mat& image);
    void setWindow(const PreviewWindow& window, bool notify = false);
    PreviewWindow window() const;
    bool hasWindow() const;
    bool isInitializedFromData() const;

signals:
    void windowChanged(double vmin, double vmax);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    enum class DragTarget
    {
        None,
        Min,
        Max
    };

    QRectF plotRect() const;
    double valueToX(double value) const;
    double xToValue(double x) const;
    void rebuildHistogram(const cv::Mat& image);
    void sanitizeWindowForData();

    std::vector<double> bins_;
    double dataMin_ = -10.0;
    double dataMax_ = 10.0;
    double vmin_ = -10.0;
    double vmax_ = 10.0;
    bool hasWindow_ = false;
    bool initializedFromData_ = false;
    DragTarget dragging_ = DragTarget::None;
};

} // namespace ct
