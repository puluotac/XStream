#pragma once

#include <QGraphicsView>

class QGraphicsPixmapItem;
class QPixmap;
class QWheelEvent;

namespace ct
{

class ZoomPanImageView : public QGraphicsView
{
    Q_OBJECT

public:
    explicit ZoomPanImageView(QWidget* parent = nullptr);

    void setImage(const QPixmap& pixmap);
    void fitWindow();

protected:
    void wheelEvent(QWheelEvent* event) override;

private:
    QGraphicsPixmapItem* pixmapItem_ = nullptr;
};

} // namespace ct
