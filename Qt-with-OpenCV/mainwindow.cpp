#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QPixmap>
#include <QDebug>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>


using namespace cv;
using namespace std;
int Minutes;
int Type;
cv::Mat starTrail;
cv::Mat resultPng;


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    qDebug() << "OpenCV Version: " << CV_VERSION;
    connect(ui->about, SIGNAL(clicked()), this, SLOT(showAbout()));
    connect(ui->discription, SIGNAL(clicked()), this, SLOT(showDiscription()));
    connect(ui->return1, SIGNAL(clicked()), this, SLOT(showHomepage()));
    connect(ui->return2, SIGNAL(clicked()), this, SLOT(showHomepage()));
    connect(ui->ok,SIGNAL(clicked()), this, SLOT(toGenerateStarTrail()));
    connect(ui->back2Home, SIGNAL(clicked()), this, SLOT(showHomepage()));
    connect(ui->startGenerate, SIGNAL(clicked()), this, SLOT(startGenerate()));
    connect(ui->pngSaveBtn, SIGNAL(clicked()), this, SLOT(savePng()));
}
MainWindow::~MainWindow()
{
    delete ui;
}



QPixmap MatToPixmap(cv::Mat src)
{
    QImage::Format format=QImage::Format_Grayscale8;
    int bpp=src.channels();
    if(bpp==3)format=QImage::Format_RGB888;
    QImage img(src.cols,src.rows,format);
    uchar *sptr,*dptr;
    int linesize=src.cols*bpp;
    for(int y=0;y<src.rows;y++){
        sptr=src.ptr(y);
        dptr=img.scanLine(y);
        memcpy(dptr,sptr,linesize);
    }
    if(bpp==3)return QPixmap::fromImage(img.rgbSwapped());
    return QPixmap::fromImage(img);
}

Mat get_star(const Mat img, Mat& star_mask, int r=174, int g=159, int b=0) {       /*-------------------生成星點------------------------*/
    Scalar lowerBound(r, g, b);
    Scalar upperBound(255, 255, 255);

    // Create a binary mask (star_mask) based on the color range
    inRange(img, lowerBound, upperBound, star_mask);

    // Apply the binary mask to the original image
    Mat star_result;
    bitwise_and(img, img, star_result, star_mask);
    cvtColor(star_result, star_result, COLOR_BGR2GRAY);

    // Find contours in the image and cancel too big contour set as 200
    vector<std::vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    Mat star_process = star_result.clone();
    dilate(star_process, star_process, cv::Mat());
    dilate(star_process, star_process, cv::Mat());
    dilate(star_process, star_process, cv::Mat());
    dilate(star_process, star_process, cv::Mat());

    findContours(star_process, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw contours on the original image
    Mat contourImage = star_result.clone();
    Mat mask = cv::Mat::ones(star_result.size(), CV_8UC1) * 255;
    //cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 255, 0), 2);
    //cout << contours.size() << endl;
    for (int i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        //cout << i << " " << area << " " << contours[i] << endl;

        if (area > 200) {
            drawContours(mask, contours, int(i), Scalar(0), -1, 8);//dst,contour,number of contour,color,fill/size of line/
            //cv::fillPoly(mask,  contours, cv::Scalar(0), 8, 0);
        }

    }
    cv::Mat resultImage;
    star_result.copyTo(resultImage, mask);
    //cv::imshow("Contour Image", resultImage);
    return resultImage;
}

Mat StarTrail(const Mat& img, Mat& result) {                             /*-------------------生成星軌------------------------*/
    Point center(img.cols / 2, img.rows / 2);
    Mat rot_mat;
    // 找最大星星
    if (Type == 0) {
        rot_mat = getRotationMatrix2D(center, 0.25, 1.0);
    }
    else if (Type == 1) {
        rot_mat = getRotationMatrix2D(center, 0.25, 0.995);
    }
    else if (Type == 2) {
        rot_mat = getRotationMatrix2D(center, 0, 0.995);
    }

    Size img_size(img.cols, img.rows);
    //int times = minutes / ;
    Mat temp;
    warpAffine(img, temp, rot_mat, img_size, INTER_LINEAR);	//轉第一下

    // 使用BORDER_CONSTANT，並指定邊界顏色為黑色
    for (int i = 0; i < Minutes; i++) {
        warpAffine(temp, temp, rot_mat, img_size, INTER_LINEAR);
        bitwise_xor(temp, result, result);
    }

    GaussianBlur(result, result, Size(3, 3), 0.2, 0.2);
    cvtColor(result, result, COLOR_GRAY2BGR);
    cout << "Star Trail's type is: " << result.type() << endl;

    Mat mask;
    threshold(result, mask, 220, 255,THRESH_BINARY_INV);
    cout << "mask's type is: " << mask.type() << endl;

    // imshow("Star Trail", result);
    // waitKey(0);
    return mask;
}

void ImageCombine(const Mat& img, const Mat& trail_mask, const Mat& trail) {         /*-------------------結合所有素材出結果----------------------*/
    cout << trail_mask.type() << endl;
    cout << img.type() << endl;
    cout << trail.type() << endl;
    bitwise_and(img, trail_mask, trail_mask);	//建立遮罩讓星軌更乾淨
    //imshow("ROI", trail_mask);
    //waitKey();


    // Mat result = Mat::zeros(img.size(), CV_8UC3);
    add(trail, trail_mask, resultPng);
    // imshow("result", resultPng);
    // waitKey();
}


/*------------------------------------------------------------------------------------------------觸發slot------------------*/
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
    QoriginImage = image;    //設定為public

    ui->stackedWidget->setCurrentIndex(3);  //換到頁面3
    // 將圖片顯示在 imageLabel 中

    QoriginImage.scaled(ui->originImage->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->originImage->setPixmap(QoriginImage);
    QoriginImage.scaled(ui->E_origin->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_origin->setPixmap(QoriginImage);


    MatImage = imread(filePath.toStdString());
    cv::resize(MatImage,MatImage,Size(MatImage.cols * 0.8, MatImage.rows * 0.8));

    Mat star_mask;
    star_result = get_star(MatImage, star_mask);
    QPixmap star = MatToPixmap(star_result);
    star.scaled(ui->starIImage->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->starIImage->setPixmap(star);

    star.scaled(ui->E_star->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_star->setPixmap(star);



}

void MainWindow::showHomepage() //回到首頁
{
    ui->stackedWidget->setCurrentIndex(0);
}
void MainWindow::showDiscription()  //顯示說明畫面
{
    ui->stackedWidget->setCurrentIndex(1);
}
void MainWindow::showAbout()    //顯示關於畫面
{
    ui->stackedWidget->setCurrentIndex(2);
}
void MainWindow::toGenerateStarTrail()
{
    ui->stackedWidget->setCurrentIndex(4);
    ui->passTime->setText("0");
    ui->E_starTrail->clear();
    ui->E_png->clear();
    ui->E_gif->clear();


}


void MainWindow::on_B_sliderMoved(int position)
{
    ui->Blabel_2->setText(QString::number(position));
    Mat star_mask;
    star_result = get_star(MatImage, star_mask,174,159,position);
    QPixmap star = MatToPixmap(star_result);
    star.scaled(ui->starIImage->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->starIImage->setPixmap(star);

    star.scaled(ui->E_star->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_star->setPixmap(star);
}


void MainWindow::on_R_sliderMoved(int position)
{
    ui->Rlabel_2->setText(QString::number(position));
    Mat star_mask;
    star_result = get_star(MatImage, star_mask,position,159,0);
    QPixmap star = MatToPixmap(star_result);
    star.scaled(ui->starIImage->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->starIImage->setPixmap(star);

    star.scaled(ui->E_star->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_star->setPixmap(star);
}


void MainWindow::on_G_sliderMoved(int position)
{
    ui->Glabel_2->setText(QString::number(position));
    Mat star_mask;
    star_result = get_star(MatImage, star_mask,174,position,0);
    QPixmap star = MatToPixmap(star_result);
    star.scaled(ui->starIImage->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->starIImage->setPixmap(star);

    star.scaled(ui->E_star->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_star->setPixmap(star);
}


void MainWindow::on_TrailLengthSlider_sliderMoved(int position)
{
    Minutes = position;
    int hour = position / 60;
    int minutes = position % 60;
    ui->TrailLength->setText("曝光時長：" + QString::number(hour) + "小時" +  QString::number(minutes) + "分鐘");
}


void MainWindow::on_TrailShape_currentIndexChanged(int index)
{
    Type = index;
}

void MainWindow::startGenerate()
{
    // ui->stackedWidget->setCurrentIndex(4);
    // ui->E_origin->setPixmap(originImage);
    double START = clock();
    Mat trail;
    star_result.copyTo(trail);
    trail.create(Size(MatImage.cols, MatImage.rows), CV_8U);
    Mat trail_mask;
    trail_mask = StarTrail(star_result, trail);

    ImageCombine(MatImage, trail_mask, trail);

    double END = clock();
    ui->passTime->setText(QString::number((END-START)/CLOCKS_PER_SEC));

    QPixmap Q_trail = MatToPixmap(trail);
    Q_trail.scaled(ui->E_starTrail->size(), Qt::KeepAspectRatio); //將圖片等比例放大至與目標label一樣大
    ui->E_starTrail->setPixmap(Q_trail);

    QPixmap Q_resultPng = MatToPixmap(resultPng);
    Q_resultPng.scaled(ui->E_png->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_png->setPixmap(Q_resultPng);
}

void MainWindow::savePng()
{
    // 打開檔案選擇視窗
    QString filePath = QFileDialog::getSaveFileName(this, "選擇路徑",  QString(), "Images (*.png *.jpg)");
    // 如果使用者取消選擇檔案，則返回
    if (!filePath.isEmpty())
        imwrite(filePath.toStdString(),resultPng);
}

