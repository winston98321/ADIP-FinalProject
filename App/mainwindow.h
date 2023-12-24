#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>

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

public slots:
    void on_uploadButton_clicked();
    void showHomepage();
    void showDiscription();
    void showAbout();
    void toGenerateStarTrail();

private slots:
    void on_B_sliderMoved(int position);

    void on_R_sliderMoved(int position);

    void on_G_sliderMoved(int position);

private:
    Ui::MainWindow *ui;

};
#endif // MAINWINDOW_H
