#include "ui/ZoomPanImageView.h"

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QFrame>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QWheelEvent>

#include <cmath>

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
    setMouseTracking(true);
    viewport()->setMouseTracking(true);
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

void ZoomPanImageView::mouseMoveEvent(QMouseEvent* event)
{
    QGraphicsView::mouseMoveEvent(event);
    if (pixmapItem_->pixmap().isNull())
    {
        emit imagePositionLeft();
        return;
    }
    const QPointF scenePoint = mapToScene(event->pos());
    const QPointF imagePoint = pixmapItem_->mapFromScene(scenePoint);
    const QRectF bounds = pixmapItem_->boundingRect();
    if (!bounds.contains(imagePoint))
    {
        emit imagePositionLeft();
        return;
    }
    emit imagePositionHovered(static_cast<int>(std::floor(imagePoint.x())),
        static_cast<int>(std::floor(imagePoint.y())));
}

void ZoomPanImageView::leaveEvent(QEvent* event)
{
    QGraphicsView::leaveEvent(event);
    emit imagePositionLeft();
}

} // namespace ct
