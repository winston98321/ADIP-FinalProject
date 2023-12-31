#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
// #include <vector>
#include <opencv2/opencv.hpp>

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

private slots:
    void on_B_sliderMoved(int position);

    void on_R_sliderMoved(int position);

    void on_G_sliderMoved(int position);

    void on_TrailLengthSlider_sliderMoved(int position);

    void on_TrailShape_currentIndexChanged(int index);

private:
    Ui::MainWindow *ui;
    QString fileName;
    // cv::Mat imgSrc;
    // std::vector<cv::UMat> img;
};
#endif // MAINWINDOW_H
