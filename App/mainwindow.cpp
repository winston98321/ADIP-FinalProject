#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QPixmap>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->about, SIGNAL(clicked()), this, SLOT(showAbout()));
    connect(ui->discription, SIGNAL(clicked()), this, SLOT(showDiscription()));
    connect(ui->return1, SIGNAL(clicked()), this, SLOT(showHomepage()));
    connect(ui->return2, SIGNAL(clicked()), this, SLOT(showHomepage()));
    connect(ui->ok,SIGNAL(clicked()), this, SLOT(toGenerateStarTrail()));
    connect(ui->back2Home, SIGNAL(clicked()), this, SLOT(showHomepage()));
}
MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_uploadButton_clicked()
{
    // QMessageBox::information(this, "Upload", "照片已上傳");
    // 打開檔案選擇視窗
    QString filePath = QFileDialog::getOpenFileName(this, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)");

    // 如果使用者取消選擇檔案，則返回
    if (filePath.isNull())
        return;

    // 顯示選擇的圖片
    QPixmap image(filePath);
    if (image.isNull()) {
        QMessageBox::warning(this, "錯誤", "無法讀取選擇的圖片");
        return;
    }
    ui->stackedWidget->setCurrentIndex(3);
    // 將圖片顯示在 imageLabel 中
    ui->originImage->setPixmap(image);
    ui->originImage->setScaledContents(true);  // 保持圖片比例
}
void MainWindow::showHomepage()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void MainWindow::showDiscription()
{
    ui->stackedWidget->setCurrentIndex(1);
}
void MainWindow::showAbout()
{
    ui->stackedWidget->setCurrentIndex(2);
}
void MainWindow::toGenerateStarTrail()
{
    ui->stackedWidget->setCurrentIndex(4);
}


void MainWindow::on_B_sliderMoved(int position)
{
    ui->Blabel_2->setText(QString::number(position));
}


void MainWindow::on_R_sliderMoved(int position)
{
    ui->Rlabel_2->setText(QString::number(position));
}


void MainWindow::on_G_sliderMoved(int position)
{
    ui->Glabel_2->setText(QString::number(position));
}

