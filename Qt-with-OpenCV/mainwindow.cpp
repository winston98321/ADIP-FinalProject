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
#include <thread>
#include <future>

using namespace cv;
using namespace std;
int Minutes;
int Type;
Point boundingBox1(0,0);
Point boundingBox2(499,499);
Point center(0,0);
Mat starTrail;
Mat resultPng;
QString InputfilePath;
bool haveFront = false;

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
    connect(ui->ok,SIGNAL(clicked()), this, SLOT(toFinalPage()));
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

Mat hsv_kmeans_seg(cv::Mat org_img, int k) {
    vector<cv::Mat> imgRGB, imgLab, imgHSV;

    cvtColor(org_img, org_img, COLOR_BGR2Lab);
    split(org_img, imgLab);

    cvtColor(org_img, org_img, COLOR_Lab2RGB);
    split(org_img, imgRGB);

    cvtColor(org_img, org_img, COLOR_RGB2HSV);
    split(org_img, imgHSV);
    cvtColor(org_img, org_img, COLOR_HSV2BGR);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

    // Set the clip limit (adjust as needed)
    clahe->setClipLimit(0.5);
    // Apply CLAHE to the input image
    cv::Mat outputImage;
    cv::Size gridSize(50, 50);  // You can change this to control the grid size
    clahe->setTilesGridSize(gridSize);

    // Apply CLAHE to the input image

    for (int i = 0; i < 3; ++i) {
        cv::medianBlur(imgLab[i], imgLab[i], 3);  // Adjust the second parameter (kernel size) as needed
        cv::medianBlur(imgRGB[i], imgRGB[i], 3);
        cv::medianBlur(imgHSV[i], imgHSV[i], 3);// Adjust the second parameter (kernel size) as needed
        blur(imgLab[i], imgLab[i], cv::Size(3, 3));
        blur(imgRGB[i], imgRGB[i], cv::Size(3, 3));
        blur(imgHSV[i], imgHSV[i], cv::Size(3, 3));
    }

    for (int i = 0; i < 3; ++i) {
        clahe->apply(imgLab[i], imgLab[i]);
        clahe->apply(imgRGB[i], imgRGB[i]);
        clahe->apply(imgHSV[i], imgHSV[i]);
    }

    int n = org_img.rows * org_img.cols;

    cv::Mat img6xN(n, 9, CV_8U);
    for (int i = 0; i < 3; i++)
        imgRGB[i].reshape(1, n).copyTo(img6xN.col(i));
    for (int i = 3; i < 6; i++)
        imgLab[i - 3].reshape(1, n).copyTo(img6xN.col(i));
    for (int i = 6; i < 9; i++)
        imgHSV[i - 6].reshape(1, n).copyTo(img6xN.col(i));
    img6xN.convertTo(img6xN, CV_32F);
    cv::Mat bestLables;
    TermCriteria criteria(TermCriteria::EPS, 100, 0.2);
    Mat labels;

    // Apply k-means algorithm

    cv::kmeans(img6xN, k, bestLables, criteria, 20, cv::KMEANS_PP_CENTERS);

    bestLables = bestLables.reshape(0, org_img.rows);
    cv::convertScaleAbs(bestLables, bestLables, int(255 / k));
    //cv::imshow("result", bestLables);

    return bestLables;
}

vector<Point> get_star_location(const cv::Mat& image) {
    vector<Point>ans;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Point p;
            if (int(image.at<uchar>(i, j)) != 0) {
                p.x = i; p.y = j;
                ans.push_back(p);
            }
        }
    }
    return ans;
}

Mat get_star(const cv::Mat img, cv::Mat star_mask, int b, int g, int r) {
    qDebug() << b <<" " << g <<" " << r;
    Scalar lowerBound(b, g, r);
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

Mat mySobel(cv::Mat image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

    // Set the clip limit (adjust as needed)
    clahe->setClipLimit(0.5);
    // Apply CLAHE to the input image
    cv::Mat outputImage;
    cv::Size gridSize(50, 50);  // You can change this to control the grid size
    clahe->setTilesGridSize(gridSize);

    // Apply CLAHE to the input image
    for (int i = 0; i < 3; ++i) {
        cv::medianBlur(channels[i], channels[i], 9);  // Adjust the second parameter (kernel size) as needed
        blur(channels[i], channels[i], cv::Size(3, 3));
        //channels[i] = guidedFilter(channels[i], channels[i], 3, 0.001);
        clahe->apply(channels[i], channels[i]);
    }

    // Apply Sobel filter to each channel
    cv::Mat sobelX, sobelY, edges2;
    cv::Mat sobelResult;

    for (int i = 0; i < 3; i++) {
        // Apply Sobel filter in X direction
        cv::Sobel(channels[i], sobelX, CV_16S, 1, 0);
        cv::Canny(channels[i], edges2, 5, 30);
        // Apply Sobel filter in Y direction
        cv::Sobel(channels[i], sobelY, CV_16S, 0, 1);

        // Convert back to 8-bit unsigned integer

        cv::convertScaleAbs(sobelX, sobelX);
        cv::convertScaleAbs(sobelY, sobelY);

        // Combine the results
        cv::addWeighted(sobelX, 0.3, sobelY, 0.7, 0, sobelResult);
        //cv::bitwise_or(sobelResult, edges, sobelResult);
        // Replace the original channel with the result
        sobelResult.copyTo(channels[i]);
    }

    // Merge the channels back into an RGB image
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8U); ; // Copy the input image to initialize the result

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            // Find the maximum value among the three channels


            uchar maxValue = std::max({ channels[0].at<uchar>(i, j), channels[1].at<uchar>(i, j), channels[2].at<uchar>(i, j) });

            // Set the result pixel value to the maximum
            result.at<uchar>(i, j) = maxValue;
        }
    }
    cv::threshold(result, result, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //cv::imshow("SobelORG Image", result);
    // Apply opening and closing operations
    cv::Mat openingResult, closingResult, closingResult2;
    //cv::dilate(star_process, star_process, cv::Mat());
    //cv::morphologyEx(star_process, openingResult, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel3);
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);
    //cv::imshow("Sobel Filtered Image", result);

    for (int col = 0; col < result.cols; col++) {
        int max_row = 9999;  // ��l�Ƴ̰�?����?

        // ���?�e�C���̰�?
        for (int row = 0; row < result.rows; row++) {
            if (static_cast<int>(result.at<uchar>(row, col)) >= 10) {
                max_row = min(row, max_row);
            }
        }

        // ?�̰�?�U��������?�m?255
        for (int row = max_row; row < result.rows; row++) {
            result.at<uchar>(row, col) = 255;
        }
    }
    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel2);
    cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel2);
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel2);

    // Display the original and filtered images
    //cv::imshow("Original Image", image);
    //cv::imshow("Sobel Filtered 255Image", result);

    return result;
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


void removeContoursTouchingImageEdges(std::vector<std::vector<cv::Point>>& contours, cv::Size imageSize) {
    std::vector<std::vector<cv::Point>> validContours;

    for (const auto& contour : contours) {
        bool isTouchingEdge = false;
        bool isupedge = false;

        for (const auto& point : contour) {
            // Check if the point is near the image edge (e.g., within a certain distance)
            if ((point.x < 5 && (point.y > (imageSize.height / 3))) || ((point.x > (imageSize.width-6)) && (point.y > (imageSize.height / 3))) || point.y > (imageSize.height-200)) {
                isTouchingEdge = true;
                break;
            }
            /*
            if (point.y < int(imageSize.height)/4) {
                //cout << "imagesize" << imageSize.height << endl;
                isupedge = true;
                break;
            }*/
        }

        if (!isTouchingEdge) {
            validContours.push_back(contour);
        }

    }

    contours = validContours;
}

Mat final_front(Mat org_img, Mat kmenas_seg) {
    Mat gray, th_re;
    cv::cvtColor(org_img, gray, COLOR_BGR2Lab);
    th_re = mySobel(gray);

    std::map<int, int> elementCount;
    std::map<int, int> frontCount;

    // �M�� th_re ���C�@�Ӥ���
    for (int i = 0; i < th_re.cols; i++) {
        for (int j = 0; j < th_re.rows; j++) {
            if (static_cast<int>(th_re.at<uchar>(i, j)) != 255) {
                int element = static_cast<int>(kmenas_seg.at<uchar>(i, j));
                elementCount[element]++;
            }
            else {
                int element = static_cast<int>(kmenas_seg.at<uchar>(i, j));
                frontCount[element]++;
            }
        }
    }
    std::vector<int> values;
    for (const auto& pair : frontCount) {
        values.push_back(pair.second);
    }
    std::sort(values.begin(), values.end(), std::greater<int>());
    int pair_min = 999999;
    for (const auto& pair : elementCount) {
        if (pair.second < pair_min) {
            pair_min = pair.second;

        }

    }

    for (const auto& pair : frontCount) {
        std::cout << "���� " << pair.first << " �X�{�F " << pair.second << " ���C" << std::endl;
    }

    for (int i = 0; i < th_re.cols; i++) {
        for (int j = 0; j < th_re.rows; j++) {
            auto it = elementCount.find(static_cast<int>(kmenas_seg.at<uchar>(i, j)));
            if (pair_min > 3000) {
                if (frontCount[static_cast<int>(kmenas_seg.at<uchar>(i, j))] < values[1] - 1) {
                    kmenas_seg.at<uchar>(i, j) = 0;
                }

                else {
                    kmenas_seg.at<uchar>(i, j) = 255;
                }

            }
            else {
                if (it != elementCount.end()) {
                    if (elementCount[static_cast<int>(kmenas_seg.at<uchar>(i, j))] > 3000) {
                        kmenas_seg.at<uchar>(i, j) = 0;
                    }
                    else {
                        kmenas_seg.at<uchar>(i, j) = 255;
                    }
                }
                else {
                    kmenas_seg.at<uchar>(i, j) = 0;

                }
            }
        }
    }
    // cv::imshow("kmenas_seg_bin", kmenas_seg);


    Mat edges;
    std::vector<std::vector<cv::Point>> contours;
    Mat grayy = kmenas_seg.clone();
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(12, 12));

    // Apply opening and closing operations
    //cv::Mat closingResult, closingResult2;
    //cv::dilate(star_process, star_process, cv::Mat());
    //cv::morphologyEx(star_process, openingResult, cv::MORPH_OPEN, kernel);
    //cv::morphologyEx(grayy, closingResult, cv::MORPH_CLOSE, kernel);
    //cv::morphologyEx(closingResult, closingResult, cv::MORPH_CLOSE, kernel);
    //cv::morphologyEx(closingResult, closingResult, cv::MORPH_CLOSE, kernel);
    //cv::imshow("closing  edges1", grayy);
    Mat process = grayy.clone();
    cv::findContours(process, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Specify the image size
    cv::Size imageSize = kmenas_seg.size();

    // Remove contours touching image edges
    removeContoursTouchingImageEdges(contours, imageSize);

    // Draw remaining contours on the original image
    cv::Mat result = grayy.clone();
    //cv::drawContours(result, contours, -1, (0, 255, 0), 5);
    //cv::imshow("closing  edges2", result);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(result, contours, int(i), Scalar(0), -1, 8);
    }
    cv::imshow("Contours without touching edges1", result);
    return result;
}


Mat StarTrail(Point center, const Mat star, Mat result) {
    //Point center(img.cols / 2, img.rows / 2);
    Mat rot_mat;
    Size img_size(star.cols, star.rows);
    if(Type == 3){
        rot_mat = getRotationMatrix2D(center, 0.5, 1.0);
        Mat temp;
        warpAffine(star, temp, rot_mat, img_size, INTER_LINEAR);	//轉第一下

        // 使用BORDER_CONSTANT，並指定邊界顏色為黑色
        for (int i = 0; i < 2 * Minutes; i++) {
            warpAffine(temp, temp, rot_mat, img_size, INTER_LINEAR);
            bitwise_or(temp, result, result);
        }

        GaussianBlur(result, result, Size(3, 3), 0.2, 0.2);
        cvtColor(result, result, COLOR_GRAY2BGR);
    }else{


        //int times = minutes / ;
        Mat temp;
        temp.setTo(0);
        threshold(star, star, 0, 255, THRESH_BINARY | THRESH_OTSU);
        // result.setTo(0);

        // 使用BORDER_CONSTANT，並指定邊界顏色為黑色

        double scale = 0.99;
        //for (int i = 0; i < minutes; i++) {
        for (double radius = 0; radius < Minutes / 60 * 15; radius += 0.5) {
            if (Type == 0) {
                rot_mat = getRotationMatrix2D(center, radius, 1.0);
            }
            else if (Type == 1) {
                rot_mat = getRotationMatrix2D(center, radius, scale);
            }
            else if (Type == 2) {
                rot_mat = getRotationMatrix2D(center, 0, scale);
            }
            warpAffine(star, temp, rot_mat, img_size);	//轉第一下
            addWeighted(result, 1.0, temp, 0.8, 0.0, result);
            scale *= 0.99;
            // imshow("temp", result);
            // waitKey();
        }

        GaussianBlur(result, result, Size(3, 3), 0.5,0.5);
        cvtColor(result, result, COLOR_GRAY2BGR);
    }
    return result;
}


Mat clahe_front(const Mat& floatImage,const Mat& front) {
    vector<cv::Mat> channels,org_channels;
    Mat org_img = floatImage.clone();
    split(floatImage, channels);
    split(org_img, org_channels);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    std::vector<cv::Mat> in_channels(channels.size());
    // Set the clip limit (adjust as needed)
    clahe->setClipLimit(0.8);
    // Apply CLAHE to the input image
    cv::Mat outputImage;
    cv::Size gridSize(12, 12);  // You can change this to control the grid size
    clahe->setTilesGridSize(gridSize);


    // ��v���@histogram equalization
    for (int i = 0; i < channels.size(); i++) {
        in_channels[i] = channels[i].clone();
        clahe->apply(in_channels[i], channels[i]);

        //histogramMapping(in_channels[i], channels[i],150,255);
    }

    for (int i = 0; i < front.rows; i++) {
        for (int j = 0; j < front.cols; j++) {
            int maskValue = static_cast<int>(front.at<uchar>(i, j));

            for (int ch = 0; ch < channels.size(); ch++) {
                if (maskValue == 0) {
                    channels[ch].at<uchar>(i,j) = org_channels[ch].at<uchar>(i,j);
                }
            }
        }
    }
    merge(channels, floatImage);
    return floatImage;
}


Mat ImageCombine(const Mat img,  const Mat trail, const Mat front) {

    Mat floatImage = img.clone();
    /*floatImage.convertTo(floatImage, CV_32F);
    normalize(floatImage, floatImage, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);*/

    //cv::imshow("CLAHE Float Image", floatImage);
    cv::Mat trail_front_result(front.size(), CV_8UC3);  // Initialize trail_front_result as a three-channel RGB image

    for (int i = 0; i < front.rows; i++) {
        for (int j = 0; j < front.cols; j++) {
            int maskValue = static_cast<int>(front.at<uchar>(i, j));

            if (maskValue > 0) {
                // Set the pixel to your desired RGB color based on your requirements
                trail_front_result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);  // Set to black (adjust as needed)
            }
            else {
                // Copy the pixel from trail to trail_front_result
                trail_front_result.at<cv::Vec3b>(i, j) = trail.at<cv::Vec3b>(i, j);
            }
        }
    }

    addWeighted(img,1.0, trail_front_result,0.5,0,img);
    //imshow("final Image", img);


    return img;
}



void make_gif(int width, int height) {
    // Convert the integers to strings
    double aspectRatio = static_cast<double>(width) / static_cast<double>(height);
    const int minSide = 512;

    int newWidth, newHeight;
    if (width < height) {
        newWidth = minSide;
        newHeight = static_cast<int>((static_cast<double>(minSide) / width) * height);
    }
    else {
        newHeight = minSide;
        newWidth = static_cast<int>((static_cast<double>(minSide) / height) * width);
    }
    std::string widthStr = std::to_string(newWidth);
    std::string heightStr = std::to_string(newHeight);
    // Concatenate the strings to form the command
    std::string command = "convert -resize " + widthStr + "x" + heightStr  +
                          " D:\\desktop\\ADIP_GIF\\*.jpg images.gif";

    // Convert the command string to a const char*
    const char* finalCommand = command.c_str();

    // Use system function to execute the command
    int result = system(finalCommand);

    // Check the execution result
    if (result == 0) {
        // Success
        std::cout << "GIF creation successful." << std::endl;
    }
    else {
        // Failure
        std::cerr << "Error: Failed to create GIF." << std::endl;
    }
}


void saveImage(const cv::Mat& image, const std::string& filename) {
    cv::imwrite(filename, image);
}

void in_thread(Point center, const cv::Mat& star_result, const cv::Mat& enhance_img, const cv::Mat& front, const std::string& filename,double time_count) {
    Mat trail_mask, final_image,trail;
    trail.create(Size(star_result.cols, star_result.rows), CV_8U);
    trail.setTo(0);
    trail_mask = StarTrail(center, star_result, trail);
    final_image = ImageCombine(enhance_img, trail, front);
    saveImage(final_image, filename);
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
    center = find_bigstar(star_result, Point(0,0), Point(499,499));   // 找最大星星
    // QRect(boundingBox1,boundingBox2);
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

void MainWindow::toFinalPage()
{
    ui->stackedWidget->setCurrentIndex(4);
    ui->passTime->setText("0");
    ui->E_starTrail->clear();
    ui->E_png->clear();
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
    resultPng.setTo(0);
    // ui->stackedWidget->setCurrentIndex(4);
    // ui->E_origin->setPixmap(originImage);


    double START = clock();
    // 取得前景
    // cv::cvtColor(resizeOriginImage, resizeOriginImage, COLOR_BGR2GRAY);


    Mat img = resizeOriginImage.clone();
    qDebug() << img.type();
    //Mat segmentedImage;    qDebug() << img.type();
    Mat segmentedImage = hsv_kmeans_seg(img, 8);
    Mat kmenas_seg = segmentedImage.clone();
    vector<Point>star_location = get_star_location(star_result);

    frontImage = final_front(img, kmenas_seg);
    QPixmap QFront =  MatToPixmap(frontImage);
    ui->E_frontandback->setPixmap(QFront);
    haveFront = true;


    // Mat enhance_img;
    enhance_img = clahe_front(resizeOriginImage, frontImage);

    // 取得星軌
    star_result.convertTo(star_result, CV_8U);
    Mat trail;
    star_result.copyTo(trail);
    trail.create(Size(resizeOriginImage.cols, resizeOriginImage.rows), CV_8U);
    trail.setTo(0);

    Mat resultTrail;
    cv::resize(star_result, star_result, Size(500, 500));

    resultTrail = StarTrail(center, star_result, trail);
    resultPng = ImageCombine(resizeOriginImage, resultTrail, frontImage);

    double END = clock();

    ui->passTime->setText(QString::number((END-START)/CLOCKS_PER_SEC));


    // show front、trail、conbine
    cv::resize(frontImage, frontImage, MatoriginImage.size());
    QPixmap QfrontImage = MatToPixmap(frontImage);
    QfrontImage.scaled(ui->E_frontandback->size(), Qt::KeepAspectRatio); //將圖片等比例放大至與目標label一樣大
    ui->E_frontandback->setPixmap(QfrontImage);

    cv::resize(resultTrail, resultTrail, MatoriginImage.size());
    QPixmap Q_trail = MatToPixmap(resultTrail);
    Q_trail.scaled(ui->E_starTrail->size(), Qt::KeepAspectRatio); //將圖片等比例放大至與目標label一樣大
    ui->E_starTrail->setPixmap(Q_trail);

    cv::resize(resultPng, resultPng, MatoriginImage.size(),0,0,INTER_AREA);
    QPixmap Q_resultPng = MatToPixmap(resultPng);
    Q_resultPng.scaled(ui->E_png->size(), Qt::KeepAspectRatio);   //將圖片等比例放大至與目標label一樣大
    ui->E_png->setPixmap(Q_resultPng);
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
    // QMessageBox::information(this, "Error", "還...還沒有...");

    double time_count = 10;
    const int numImages = time_count;

    // 設定儲存的檔案名稱
    std::vector<std::string> filenames;
    for (int i = 0; i < numImages; i++) {
        filenames.push_back("gif/" + std::to_string(i) + ".jpg");
    }

    // 使用 std::async 同時儲存多張圖片
    //in_thread(center, star_result,  enhance_img, front, filenames[0], double(time_count / 30));

    std::vector<std::future<void>> futures;
    for (int i = 0; i < numImages; i++) {
        futures.push_back(std::async(std::launch::async, [=]() {
            in_thread(center, star_result,  enhance_img, frontImage, filenames[i], double(time_count/30));
        }));
    }

    int w = resizeOriginImage.cols; int h = resizeOriginImage.rows;
    make_gif(w, h);

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
    center = find_bigstar(star_result, boundingBox1, boundingBox2);   // 找最大星星
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












