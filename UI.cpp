//112318155許楷暄&112318003蔡尚哲
//第一週 第七組code
//指導老師李曉祺

/*
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <vector>
#include <stack>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <opencv2/videoio.hpp>
using namespace cv;
using namespace std;

Mat image;
Point start;
Point end1;// 記錄矩形選區的起點和終點
Point centroidd;
bool drawing = false;
void onMouse(int event, int x, int y, int flags, void* param); 

vector<vector<Point>> star;
vector<vector<Point>> contours;
vector<Mat> channels;
vector<cv::Mat> resultVector;
float ss = 1000 * 1000;




int main() {
    string dark_path = "D:/Final project Dataset/dark foreground/foreground_1.jpg";
    string aurora_path = "D:/Final project Dataset/aurora/aurora_test1.jpg";
    string light_path = "D:/Final project Dataset/light foreground/light_foreground_5.jpg";
    string test_path = ".jpg";
    char filename[100];
    cout << "Please enter your filename" << endl;
    cin.getline(filename, 100);
    image = imread(light_path);
    
    //decalare kernel
    int radius1 = 1.5;  // 設定半徑
    Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * radius1 + 1, 2 * radius1 + 1));
    int radius2 = 0.0;
    Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(2 * radius2 + 1, 2 * radius2 + 1));
    int height = image.rows;
    int width = image.cols;
    
    float size = height * width;
    height = height / sqrt(size / ss);
    width = width / sqrt(size / ss);
    int a;
    printf("which case(1. circle, 2.sprial, 3. scattering): ");
    scanf("%d", &a);
    printf("wait a minute....\n");
    resize(image, image, Size(width, height));
  
    
    // 將圖片轉換為灰度

    rectangle(image, Rect(0, 0, width, height), Scalar(255, 255, 255), 1);
  
    imshow("orgin image", image);
    setMouseCallback("orgin image", onMouse, &image);
    waitKey(0);
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    //equalizeHist(grayImage, grayImage);
    GaussianBlur(grayImage, grayImage, Size(11, 11), 0);//對edge做第一次模糊
    
    Mat edges;
    Canny(grayImage, edges, 10, 20);//沒有normallize
    morphologyEx(edges, edges, MORPH_DILATE, kernel1);
    GaussianBlur(edges, edges, Size(7, 7), 0);
    // 找到輪廓

    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    drawContours(mask, contours, -1, Scalar(255), FILLED);

    // 前後景ROI製作
    Mat fore_mask;
    int count = 0;
    for (int i = contours.size() - 1; i > 0; --i) {
        // 創建一個新的掩碼，用白色填充當前輪廓

        double area = contourArea(contours[i]);
        double areaThreshold = 500;

        if (area >= areaThreshold) {
            count++;
            if (count == 2) {
                // 創建一個新的掩碼，用白色填充當前輪廓
                Mat contour_mask = Mat::zeros(image.size(), CV_8UC1);
                drawContours(contour_mask, contours, static_cast<int>(i), Scalar(255), FILLED);

                // 將原始圖像中輪廓區域以外的部分視為背景，使用原始圖像的背景色填充
                Scalar bgColor = mean(image, contour_mask);
                Mat fore_temp = Mat::zeros(image.size(), CV_8UC1);
                fore_temp.setTo(Scalar(0), ~contour_mask);
                fore_temp.setTo(Scalar(255), contour_mask);
                // 製作只有0&1的mask
                fore_mask = Mat::zeros(fore_temp.size(), CV_8UC1);
                for (int j = 0; j < fore_temp.rows; j++) {
                    for (int k = 0; k < fore_temp.cols; k++) {
                        if (int(fore_temp.at<uchar>(j, k)) != 0) {
                            fore_mask.at<uchar>(j, k) = 1;
                        }
                    }
                }
                imwrite("Contour " + std::to_string(i), fore_mask * 255);  // 顯示結果時將圖像轉回 0-1 的範圍
            }
        }

    }

    //--------------------找星星-------------------------------------//
   
    Mat graystar;
    cvtColor(image, graystar, COLOR_BGR2HSV);



    split(graystar, channels);

    Mat edges_star;
    Mat brightnessChannel = channels[2];
    ////imshow("Edge star0", brightnessChannel);
    Mat thresholdImage;
    //-------------乘遮照-------------------------------------------//
    multiply(brightnessChannel, fore_mask, edges_star);
    Canny(edges_star, edges_star, 60, 120);//沒有normallize
    rectangle(edges_star, Rect(0, 0, width, height), Scalar(0), 12);
    morphologyEx(edges_star, edges_star, MORPH_DILATE, kernel1);

    Mat kernel3 = getStructuringElement(MORPH_CROSS, Size(2, 2));
    Mat kernel4 = getStructuringElement(MORPH_CROSS, Size(2, 2));

    //morphologyEx(edges_star, edges_star, MORPH_ERODE, kernel4);
    morphologyEx(edges_star, edges_star, MORPH_ERODE, kernel3);
    morphologyEx(edges_star, edges_star, MORPH_DILATE, kernel2);
    //imshow("Edge star1", edges_star);


    findContours(edges_star, star, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    //星星遮罩宣告
    Mat sttar = Mat::zeros(image.size(), CV_8UC1);  
    drawContours(edges_star, star, -1, Scalar(255), 1);
    double maxBrightness = -1;
    int maxBrightnessContourIndex = -1;
    multiply(edges_star, fore_mask, edges_star);

    for (int i = 0; i < star.size(); ++i) {
        // 計算輪廓的面積
        double area1 = contourArea(star[i]);
        if (area1 <= 300) {
            Mat mask1 = Mat::zeros(graystar.size(), CV_8UC1);
            drawContours(mask1, star, i, Scalar(255), FILLED);
            double meanBrightness = mean(brightnessChannel, mask1)[0];

            Moments moments = cv::moments(star[i]);

            // 計算中心點
            Point centroid(static_cast<int>(moments.m10 / moments.m00), static_cast<int>(moments.m01 / moments.m00));

            // 繪製中心點
            int radius = static_cast<int>(sqrt(area1 / CV_PI));
            circle(sttar, centroid, radius, Scalar(255), -1);
            // 如果亮度較大，更新最大亮度和最大亮度的輪廓索引
            if (meanBrightness > maxBrightness) {
                maxBrightness = meanBrightness;
                maxBrightnessContourIndex = i;
            }

        }
        // 計算輪廓內的平均亮度

    }
    Point centroid;
    if (maxBrightnessContourIndex != -1) {
        cv::Moments moments = cv::moments(star[maxBrightnessContourIndex]);
        centroid = Point2f(static_cast<int>(moments.m10 / moments.m00), static_cast<int>(moments.m01 / moments.m00));
        //std::cout << "Max Brightness Star Center: (" << centroid.x << ", " << centroid.y << ")" << std::endl;
    }

    Mat result1 = image.clone();
    drawContours(result1, star, maxBrightnessContourIndex, Scalar(0, 255, 0), 2);
    imshow("Max bright star", result1);
    multiply(sttar, fore_mask, sttar);

  
    Mat star_mask;
    star_mask = Mat::zeros(sttar.size(), CV_8UC1);

    for (int j = 0; j < sttar.rows; j++) {
        for (int k = 0; k < sttar.cols; k++) {
            if (int(sttar.at<uchar>(j, k)) != 0) {
                star_mask.at<uchar>(j, k) = 1;
            }
        }
    }

 

    vector<Mat> channels;
    split(image, channels);
    // 顯示結果
    //imshow("Bright Areas and Centroids (HSI)", sttar);
    for (int i = 0; i < 3; ++i) {
        channels[i].convertTo(channels[i], CV_8UC1);
        multiply(star_mask, channels[i], channels[i]);
        normalize(channels[i], channels[i], 0, 255, NORM_MINMAX);
    }
    Mat sttar_RGB;
   
    Point2f center(centroidd.x,centroidd.y);
    cout << "The center u choose:(" << centroidd.x << ", " << centroidd.y << " )" << endl;
    merge(channels, sttar_RGB);
    //imshow("star_mask", sttar_RGB);
    

    circle(sttar_RGB, center, 40, Scalar(0, 0, 0), -1);
    
    ////imshow("Bright Areas and Centroids (HSI)", sttar_RGB);
    // 
    //螺旋&散射
    sttar_RGB.convertTo(sttar_RGB, CV_8UC3);
    Mat scattering;
    scattering = Mat::zeros(image.size(), CV_8UC3);
    Mat temp;
    temp = Mat::zeros(image.size(), CV_8UC3);
    Mat result;
    image.copyTo(result);
    Mat scattering_temp;

    //VideoWriter writer("output.gif",VideoWriter::fourcc('G', 'I', 'F', 'S'), 10, result.size());
    for (double radius = 0; radius < 100; radius += 2) {
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double x = j - center.x;
                double y = i - center.y;
                double d = sqrt(x * x + y * y);
                int NewX;
                int NewY;
            
            if(a==1){
                NewX = j;//circle
                NewY = i;
            }
            else {
                NewX = j + (x * radius / d);  //scattering or sprial
                NewY = i + (y * radius / d);
            }

                if (NewX >= 0 && NewX < sttar.cols && NewY >= 0 && NewY < sttar.rows) {
                    temp.at<Vec3b>(NewY, NewX) = sttar_RGB.at<Vec3b>(i, j);
                    //std::cout << "Invalid indices: NewX = " << NewX << ", NewY = " << NewY << std::endl;
                }


            }
        }
        Mat rotationMatrix = getRotationMatrix2D(center, radius * CV_PI * 10 / 180, 1.0);
        if (a != 3){ 
            warpAffine(temp, temp, rotationMatrix, scattering.size()); 
        }
         addWeighted(scattering, 1.0, temp,0.5 , 0.0, scattering);
         split(scattering, channels);
         for (int i = 0; i < 3; ++i) {
             channels[i].convertTo(channels[i], CV_8UC1);
             multiply(fore_mask, channels[i], channels[i]);
             //normalize(channels[i], channels[i], 0, 255, NORM_MINMAX);
         }
         merge(channels, scattering);


         addWeighted(result, 1.0, scattering, -1.0, 0.0, result);
         addWeighted(result, 1.0, scattering, 1.2, 0.0, result);
         resultVector.push_back(result.clone());
         

     cv::imshow("result", result);
     cv::waitKey(1);
    }

   if (!resultVector.empty()) {
       if (a == 1) {
           cv::VideoWriter writer("select_center_circle.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, resultVector[0].size());
           for (const auto& frame : resultVector) {
               writer.write(frame);
           }
           writer.release();
       }
       else if (a == 2) {
           cv::VideoWriter writer("select_center_sprial.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, resultVector[0].size());
           for (const auto& frame : resultVector) {
               writer.write(frame);
           }
           writer.release();
       }
       else if (a == 3) {
           cv::VideoWriter writer("select_center_scattering.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, resultVector[0].size());
           for (const auto& frame : resultVector) {
               writer.write(frame);
           }
           writer.release();
       }
   }

   imshow("Star", sttar_RGB);
   imshow("result", result);
   if (a == 1) {
       imwrite("select_center_circle.png", result);
       waitKey(0);
   }
   else if (a == 2) {
       imwrite("select_center_sprial.png", result);
       waitKey(0);
   }
   else if (a == 3) {
       imwrite("select_center_scattering.png", result);
       waitKey(0);
   }
    return 0;
}

void onMouse(int event, int x, int y, int flags, void* param) {
    static Mat temp_image;
    image.copyTo(temp_image);

    if (event == EVENT_LBUTTONDOWN) {
        start = Point(x, y);
    }
    else if (event == EVENT_LBUTTONUP) {
        end1 = Point(x, y);

        // 計算選區的中心點
        centroidd.x = (start.x + end1.x) / 2;
        centroidd.y = (start.y + end1.y) / 2;
        rectangle(temp_image, start, end1, Scalar(255, 0, 0), 2);
        imshow("orgin image", temp_image);
        
    }
    
}*/