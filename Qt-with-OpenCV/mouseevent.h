#ifndef MOUSEEVENT_H
#define MOUSEEVENT_H
#include <QLABEL>
#include <QMouseEvent>
#include <QEvent>
#include <QDebug>
class MouseEvent : public QLabel
{
    Q_OBJECT
public:
    explicit MouseEvent(QWidget *parent = nullptr);
    void mouseMoveEvent(QMouseEvent *ev) override;
    void mousePressEvent(QMouseEvent *ev) override;
    void mouseReleaseEvent(QMouseEvent *ev) override;
    void leaveEvent(QEvent *);
    int x, y;

signals:
    void Mouse_Pressed();
    void Mouse_Release();
    void Mouse_Move();
    void Mouse_Left();
};

#endif // MOUSEEVENT_H
