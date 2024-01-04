#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QMouseEvent>
#include <opencv2/opencv.hpp>
#include <QLabel>
#include <QPainter>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    QPixmap QoriginImage;
    cv::Mat MatImage;
    cv::Mat star_result;




public slots:
    void on_uploadButton_clicked();
    void showHomepage();
    void showDiscription();
    void showAbout();
    void toGenerateStarTrail();
    void startGenerate();
    void savePng();
    void saveGif();

    void Mouse_Pressed();
    void Mouse_Relese();
    void Mouse_Move();
    void Mouse_Left();

private slots:
    void on_B_sliderMoved(int position);

    void on_R_sliderMoved(int position);

    void on_G_sliderMoved(int position);

    void on_TrailLengthSlider_sliderMoved(int position);

    void on_TrailShape_currentIndexChanged(int index);

    void on_R_valueChanged(int value);

    void on_G_valueChanged(int value);

    void on_B_valueChanged(int value);

private:
    Ui::MainWindow *ui;
    QString fileName;
    bool selecting;
    QPoint selectionStart;
    QPoint selectionEnd;
};

#endif // MAINWINDOW_H
