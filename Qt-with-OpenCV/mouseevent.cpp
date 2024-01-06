#include "mouseevent.h"

MouseEvent::MouseEvent(QWidget *parent): QLabel(parent)
{

}
void MouseEvent::mouseMoveEvent(QMouseEvent *ev)
{
    // get current mouse position
    this->x = ev->x();
    this->y = ev->y();
    emit Mouse_Move();

    // qDebug() << "signal:: mouseMoveEvent";
}
void MouseEvent::mousePressEvent(QMouseEvent *ev)
{
    this->position = ev->pos();
    emit Mouse_Pressed();
    // qDebug() << "signal:: mousePressEvent";

}

void MouseEvent::mouseReleaseEvent(QMouseEvent *ev)
{
    this->position = ev->pos();
    emit Mouse_Release();
    // qDebug() << "signal:: mouseReleaseEvent";
}

void MouseEvent::leaveEvent(QEvent *)
{
    emit Mouse_Left();
    // qDebug() << "signal::leaveEvent";
}
