#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QPixmap>
#include <QDebug>
#include <QMouseEvent>
#include <QPainter>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <cstdlib>

using namespace cv;
using namespace std;
int Minutes;
int Type;
Point boundingBox1(0,0);
Point boundingBox2(0,0);
Point center(0,0);
Mat starTrail;
Mat resultPng;
QString InputfilePath;


MainWindow::MainWindow(QWidget *parent):
    QMainWindow(parent)
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
    connect(ui->gifSaveBtn, SIGNAL(clicked()), this, SLOT(saveGif()));

    connect(ui->starIImage, SIGNAL(Mouse_Move()), this, SLOT(Mouse_Move()));
    connect(ui->starIImage, SIGNAL(Mouse_Pressed()), this, SLOT(Mouse_Pressed()));
    connect(ui->starIImage, SIGNAL(Mouse_Release()), this, SLOT(Mouse_Relese()));
    connect(ui->starIImage, SIGNAL(Mouse_Left()), this, SLOT(Mouse_Left()));
}


MainWindow::~MainWindow()
{
    delete ui;
}



QPixmap MatToPixmap(Mat src)
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

Mat get_star(const cv::Mat img, cv::Mat star_mask, int r, int g, int b) {
    qDebug() << r <<" " << g <<" " << b;
    Scalar lowerBound(174, 159, 0);
    Scalar upperBound(255, 255, 255);

    // Create a binary mask (star_mask) based on the color range
    inRange(img, lowerBound, upperBound, star_mask);

    // Apply the binary mask to the original image
    Mat star_result;
    bitwise_and(img, img, star_result, star_mask);
    cvtColor(star_result, star_result, COLOR_BGR2GRAY);

    // cv::imshow("Star Mask", star_result);

    // Find contours in the image and cancel too big contour set as 200
    vector<std::vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    Mat star_process = star_result.clone();
    Mat final = star_process.clone();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

    // Apply opening and closing operations
    cv::Mat openingResult, closingResult, closingResult2;
    //cv::dilate(star_process, star_process, cv::Mat());
    //cv::morphologyEx(star_process, openingResult, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(star_process, closingResult, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(closingResult, closingResult2, cv::MORPH_CLOSE, kernel);

    Mat contourImage = closingResult2.clone();
    findContours(closingResult2, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw contours on the original image

    Mat mask = cv::Mat::ones(star_result.size(), CV_8UC1) * 255;
    //cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 255, 0), 2);
    //cout << contours.size() << endl;
    for (int i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        //cout << i << " " << area << " " << contours[i] << endl;

        if (area > 150) {
            drawContours(mask, contours, int(i), Scalar(0), -1, 8);//dst,contour,number of contour,color,fill/size of line/
            //cv::fillPoly(mask,  contours, cv::Scalar(0), 8, 0);
        }

    }
    cv::Mat resultImage;
    star_result.copyTo(resultImage, mask);
    //cv::imshow("Contour Image", resultImage);
    Mat threshold_resultImage;
    cv::threshold(resultImage, resultImage, 10, 255, cv::THRESH_BINARY);

    return resultImage;
}


Point find_bigstar(Mat star_result, Point left_up, Point right_down) {
    Mat result = star_result.clone();

    // Canny(star_result, result, 5, 30);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    Mat find_contour = result.clone();
    cv::Rect ROI(left_up.x, left_up.y, right_down.x - left_up.x, right_down.y - left_up.y);
    cv::Mat croppedImage = find_contour(ROI);
    cv::findContours(croppedImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double area = -1;
    int index = -1;

    for (int i = 0; i < contours.size(); i++) {
        //cout << i << " " << area << " " << contours[i] << endl;
        if (cv::contourArea(contours[i]) > area) {
            area = contourArea(contours[i]);
            index = i;
        }
    }

    Point center(-1, -1);

    if (index >= 0) {

        Moments mu = moments(contours[index]);
        center = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
        center.x += left_up.x;
        center.y += left_up.y;
    }
    //如果沒找到直接回傳圖片中最大星星位置
    else {
        cv::findContours(find_contour, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double area = -1;
        int index = -1;

        for (int i = 0; i < contours.size(); i++) {
            //cout << i << " " << area << " " << contours[i] << endl;
            if (cv::contourArea(contours[i]) > area) {
                area = contourArea(contours[i]);
                index = i;
            }
        }

        if (index >= 0) {
            Moments mu = moments(contours[index]);
            center = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
            center.x += left_up.x;
            center.y += left_up.y;
        }
    }
    return center;
}


Mat StarTrail(Point center, const Mat& img, Mat& result) {                             /*-------------------生成星軌------------------------*/
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
        bitwise_or(temp, result, result);
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


void make_gif() {
    const char* command = "magick gif/*.jpg images.gif";
    // 使用 system 函數執行命令
    int result = system(command);
    // 檢查執行結果
    if (result == 0) {
        // 成功
        std::cout << "GIF creation successful." << std::endl;
    }
    else {
        // 失敗
        std::cout << "Error: Failed to create GIF." << std::endl;
    }
    system("pause");
}

/*------------------------------------------------------------------------------------------------觸發slot------------------*/
void MainWindow::on_uploadButton_clicked()
{
    // 打開檔案選擇視窗
    InputfilePath = QFileDialog::getOpenFileName(this, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)");
    // 如果使用者取消選擇檔案，則返回
    if (InputfilePath.isNull())
        return;

    // 顯示選擇的圖片
    QPixmap image(InputfilePath);
    if (image.isNull()) {
        QMessageBox::warning(this, "錯誤", "無法讀取選擇的圖片");
        return;
    }
    QoriginImage = image;    //設定為public

    ui->stackedWidget->setCurrentIndex(3);  //換到頁面3
    // 將圖片顯示在 imageLabel 中


    ui->originImage->setScaledContents(true);
    QoriginImage.scaled(ui->originImage->size(),Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);    //將圖片等比例放大至與目標label一樣大
    ui->originImage->setPixmap(QoriginImage);


    QoriginImage.scaled(ui->E_origin->size(),Qt::KeepAspectRatio, Qt::SmoothTransformation);        //將圖片等比例放大至與目標label一樣大
    ui->E_origin->setPixmap(QoriginImage);

    MatoriginImage = imread(InputfilePath.toStdString());
    // cv::resize(MatImage,MatImage,Size(MatImage.cols * 0.8, MatImage.rows * 0.8));
    cv::resize(MatoriginImage,resizeOriginImage,Size(500, 500));

    Mat star_mask;
    star_result = get_star(resizeOriginImage, star_mask,174, 154, 60);  // 取得所有星星
    center = find_bigstar(star_result, Point(0, 0), Point(500, 500));   // 找最大星星
    cout << "Big Star Location is: " << center << endl;

    cv::resize(star_result, star_result, MatoriginImage.size());

    QPixmap star = MatToPixmap(star_result);
    star.scaled(ui->starIImage->size(), Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);   //將圖片等比例放大至與目標label一樣大
    ui->starIImage->setPixmap(star);
    star.scaled(ui->E_star->size(), Qt::KeepAspectRatio);       //將圖片等比例放大至與目標label一樣大
    ui->E_star->setPixmap(star);

}

void MainWindow::showHomepage() //回到首頁
{
    ui->stackedWidget->setCurrentIndex(0);

    ui->R->setValue(174);
    ui->Rlabel_2->setText(QString::number(174));
    ui->G->setValue(154);
    ui->Glabel_2->setText(QString::number(154));
    ui->B->setValue(60);
    ui->Blabel_2->setText(QString::number(60));

    ui->TrailLengthSlider->setValue(0);
    ui->TrailLength->setText("曝光時長：" + QString::number(0) + "小時" +  QString::number(0) + "分鐘");
    ui->TrailShape->setCurrentIndex(0);
}

void MainWindow::showDiscription()  //顯示說明畫面
{
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::showAbout()    //顯示關於畫面
{
    ui->stackedWidget->setCurrentIndex(2);
}
/*--------------------------------------------------------------------------------------------RGB頁設定OK 生成mask 進入下一步---------*/

void MainWindow::toGenerateStarTrail()
{
    ui->stackedWidget->setCurrentIndex(4);
    ui->passTime->setText("0");
    ui->E_starTrail->clear();
    ui->E_png->clear();
    ui->E_gif->clear();

    // segmentedImage前景分割;
    // Mat segmentedImage = hsv_kmeans_seg(resizeOriginImage,8);
    // cv::resize(star_result, star_result, Size(500,500));
    // vector<Point>star_location = get_star_location(star_result);
    // //cv::cvtColor(org_img, org_img, COLOR_Lab2BGR);
    // cv::imshow("segmentedImage_result", segmentedImage);
    // medianBlur(segmentedImage, segmentedImage, 3);
    // for (int i = 0; i < star_location.size(); i++) {
    //     cout << star_location[i].x << " " << star_location[i].y << endl;
    //     Mat mymask = customFloodFill(segmentedImage, int(star_location[i].y), int(star_location[i].x), 0);
    // }
    // imshow("segmentedImage_result2", segmentedImage);
    // waitKey();
}



/*----------------------------------------------------------------------------------------------------------RGB slider更改---------*/
void MainWindow::on_R_sliderMoved(int position)
{
    ui->Rlabel_2->setText(QString::number(position));

}

void MainWindow::on_R_valueChanged(int value)
{
    Mat star_mask;
    int g = ui->G->value();
    int b = ui->B->value();
    qDebug() << "r = " << value;

    star_result = get_star(resizeOriginImage, star_mask,value,g,b);
    cv::resize(star_result, star_result, MatoriginImage.size(),0,0,INTER_AREA);
    QPixmap star = MatToPixmap(star_result);
    star.scaled(ui->starIImage->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->starIImage->setPixmap(star);

    star.scaled(ui->E_star->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_star->setPixmap(star);
}


void MainWindow::on_G_sliderMoved(int position)
{
    ui->Glabel_2->setText(QString::number(position));

}
void MainWindow::on_G_valueChanged(int value)
{
    Mat star_mask;
    int r = ui->R->value();
    int b = ui->B->value();
    star_result = get_star(resizeOriginImage, star_mask,r,value,b);
    cv::resize(star_result, star_result, MatoriginImage.size());
    QPixmap star = MatToPixmap(star_result);
    star.scaled(ui->starIImage->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->starIImage->setPixmap(star);

    star.scaled(ui->E_star->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_star->setPixmap(star);
}


void MainWindow::on_B_sliderMoved(int position)
{
    ui->Blabel_2->setText(QString::number(position));

}

void MainWindow::on_B_valueChanged(int value)
{
    Mat star_mask;
    int r = ui->R->value();
    int g = ui->G->value();

    star_result = get_star(resizeOriginImage, star_mask,r,g,value);
    cv::resize(star_result, star_result, MatoriginImage.size());
    QPixmap star = MatToPixmap(star_result);
    star.scaled(ui->starIImage->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->starIImage->setPixmap(star);

    star.scaled(ui->E_star->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_star->setPixmap(star);
}


/*----------------------------------------------------------------------------------------------------------星軌長度---------*/

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


/*----------------------------------------------------------------------------------------------------------開始生成總結果---------*/
void MainWindow::startGenerate()
{
    // ui->stackedWidget->setCurrentIndex(4);
    // ui->E_origin->setPixmap(originImage);
    double START = clock();
    Mat trail;
    cv::resize(star_result, star_result, Size(500,500));
    star_result.copyTo(trail);
    trail.create(Size(resizeOriginImage.cols, resizeOriginImage.rows), CV_8U);
    Mat trail_mask;
    trail_mask = StarTrail(center, star_result, trail);
    ImageCombine(resizeOriginImage, trail_mask, trail);


    double END = clock();
    ui->passTime->setText(QString::number((END-START)/CLOCKS_PER_SEC));

    cv::resize(trail, trail, MatoriginImage.size());
    QPixmap Q_trail = MatToPixmap(trail);
    Q_trail.scaled(ui->E_starTrail->size(), Qt::KeepAspectRatio); //將圖片等比例放大至與目標label一樣大
    ui->E_starTrail->setPixmap(Q_trail);

    cv::resize(resultPng, resultPng, MatoriginImage.size(),0,0,INTER_AREA);
    QPixmap Q_resultPng = MatToPixmap(resultPng);
    Q_resultPng.scaled(ui->E_png->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_png->setPixmap(Q_resultPng);

    // cv::resize(resultPng,resultPng,Size(resultPng.cols / 0.8, resultPng.rows / 0.8));
}

void MainWindow::savePng()
{
    cv::resize(resultPng,resultPng, MatoriginImage.size());

    // 打開檔案選擇視窗
    QString filePath = QFileDialog::getSaveFileName(this, "選擇路徑",  InputfilePath, "Images (*.png *.jpg)");
    // 如果使用者取消選擇檔案，則返回
    if (!filePath.isEmpty())
        imwrite(filePath.toStdString(),resultPng);
}


void MainWindow::saveGif()
{
    QMessageBox::information(this, "Error", "還...還沒有...");
    make_gif();

    // // 打開檔案選擇視窗
    // QString filePath = QFileDialog::getSaveFileName(this, "選擇路徑",  InputfilePath, "Images (*.gif *.avi)");
    // // 如果使用者取消選擇檔案，則返回
    // if (!filePath.isEmpty())
    //     imwrite(filePath.toStdString(),resultGif);
}

void MainWindow::Mouse_Pressed()
{
    ui->lblMouse_Current_Event->setText("Mouse Pressed!");
    qDebug() << "ui slot:: Mouse Pressed!";
    int xx = ui->starIImage->position.x();
    int yy = ui->starIImage->position.y();
    boundingBox1.x = xx;
    boundingBox1.y = yy;
}

void MainWindow::Mouse_Relese()
{
    ui->lblMouse_Current_Event->setText("Mouse Released!");
    qDebug() << "ui slot:: Mouse Released!";
    int xx = ui->starIImage->position.x();
    int yy = ui->starIImage->position.y();
    boundingBox2.x = xx;
    boundingBox2.y = yy;
}

void MainWindow::Mouse_Move()
{
    ui->lblMouse_Current_Pos->setText(QString("X = %1, Y = %2").arg(ui->starIImage->x).arg(ui->starIImage->y));
    ui->lblMouse_Current_Event->setText("Mouse Moving!");
}


void MainWindow::Mouse_Left()
{
    ui->lblMouse_Current_Event->setText("Mouse Left!");
    qDebug() << "ui slot:: Mouse Left!";
}












