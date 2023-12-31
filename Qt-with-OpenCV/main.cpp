#include "mainwindow.h"
#include<opencv2/opencv.hpp>
#include <QApplication>
#include<iostream>
#undef slots

#define slots Q_SLOTS


int main(int argc, char *argv[])
{

    //test opencv
    // cv::Mat image = cv::imread("D:\\desktop\\11201.png");
    // cv::Mat M(200, 200, CV_8UC3, cv::Scalar(0, 0, 255));
    // if(!M.data)
    //     return 0;
    // cv::imshow("fff",image);
    // cv::imshow("ddd",M);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    //test qt
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
