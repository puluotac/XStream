#include "ui/ZoomPanImageView.h"

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QFrame>
#include <QPainter>
#include <QPixmap>
#include <QWheelEvent>

namespace ct
{

ZoomPanImageView::ZoomPanImageView(QWidget* parent)
    : QGraphicsView(parent)
{
    auto* scene = new QGraphicsScene(this);
    setScene(scene);

    pixmapItem_ = new QGraphicsPixmapItem();
    scene->addItem(pixmapItem_);

    setRenderHint(QPainter::Antialiasing, false);
    setRenderHint(QPainter::SmoothPixmapTransform, false);
    setOptimizationFlag(QGraphicsView::DontAdjustForAntialiasing, true);
    setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorUnderMouse);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setBackgroundBrush(Qt::black);
    setFrameShape(QFrame::NoFrame);
}

void ZoomPanImageView::setImage(const QPixmap& pixmap)
{
    pixmapItem_->setPixmap(pixmap);
    scene()->setSceneRect(pixmapItem_->boundingRect());
}

void ZoomPanImageView::fitWindow()
{
    if (pixmapItem_->pixmap().isNull())
    {
        return;
    }
    fitInView(pixmapItem_, Qt::KeepAspectRatio);
}

void ZoomPanImageView::wheelEvent(QWheelEvent* event)
{
    constexpr double zoomInFactor = 1.15;
    constexpr double zoomOutFactor = 1.0 / zoomInFactor;
    scale(event->angleDelta().y() > 0 ? zoomInFactor : zoomOutFactor,
        event->angleDelta().y() > 0 ? zoomInFactor : zoomOutFactor);
}

} // namespace ct
