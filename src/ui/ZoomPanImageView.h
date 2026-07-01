#pragma once

#include <QGraphicsView>

class QGraphicsPixmapItem;
class QEvent;
class QPixmap;
class QWheelEvent;
class QMouseEvent;

namespace ct
{

class ZoomPanImageView : public QGraphicsView
{
    Q_OBJECT

public:
    explicit ZoomPanImageView(QWidget* parent = nullptr);

    void setImage(const QPixmap& pixmap);
    void fitWindow();

signals:
    void imagePositionHovered(int x, int y);
    void imagePositionLeft();

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    QGraphicsPixmapItem* pixmapItem_ = nullptr;
};

} // namespace ct
