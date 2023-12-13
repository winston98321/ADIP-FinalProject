#include <opencv2/opencv.hpp>
#include <string>
#include <Vector>

using namespace cv;
using namespace std;

void toNegative(Mat& input, Mat& output) {
	int width = input.cols;
	int height = input.rows;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output.at<Vec3b>(i, j) = Vec3b(255, 255, 255) - input.at<Vec3b>(i, j);
		}
	}
}


void imgLog(Mat& input, Mat& output) {
	int width = input.cols;
	int height = input.rows;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int z = 0; z < input.channels(); z++) {
				output.at<Vec3b>(i, j)[z] = log(1 + int(input.at<Vec3b>(i, j)[z])) / log(256) * 255;
				if (int(output.at<Vec3b>(i, j)[z]) > 255)
					output.at<Vec3b>(i, j)[z] = 255;
				else if (int(output.at<Vec3b>(i, j)[z]) < 0)
					output.at<Vec3b>(i, j)[z] = 0;
			}
		}
	}
}

void histogram(Mat& input) {
	vector<Mat> bgr_planes;
	split(input, bgr_planes);
	int histSize = 256;

	float range[] = { 0, 256 }; //the upper boundary is exclusive
	int width = input.cols;
	int height = input.rows;
	bool uniform = true, accumulate = false;
	const float* histRange[] = { range };

	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());


	int maxB = 0;
	int maxG = 0;
	int maxR = 0;
	int indexB, indexG, indexR;
	cout << (b_hist) << endl;
	for (int i = 1; i < histSize; i++)
	{
		if (b_hist.at<float>(i) > maxB) {
			maxB = b_hist.at<float>(i);
			indexB = i;
		}

		if (g_hist.at<float>(i) > maxG) {
			maxG = g_hist.at<float>(i);
			indexG = i;
		}

		if (r_hist.at<float>(i) > maxR) {
			maxR = r_hist.at<float>(i);
			indexR = i;
		}

	}
	cout << "The max of B is " << to_string(maxB) << endl;
	cout << "The max of G is " << to_string(maxG) << endl;
	cout << "The max of R is " << to_string(maxR) << endl;

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("calcHist Demo", histImage);
	waitKey();
}



int main()
{
	string fileName = "./img./aurora_2.jpg";
	Mat img = imread(fileName);
	if (img.cols > 1920)
		resize(img, img, Size(img.cols / 8, img.rows / 8));

	imshow("open", img);
	waitKey();

	//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(12, 12));
	//morphologyEx(img, img, MORPH_OPEN, element);
	//Mat negativeImg = Mat::zeros(img.size(), img.type());
	//toNegative(img, negativeImg);
	//Mat LogImg = Mat::zeros(img.size(), img.type());

	////imgLog(img, LogImg);
	//histogram(img);

	cvtColor(img, img, COLOR_BGR2GRAY);
	//Sobel(img, img, CV_8U, 0,1);
	Canny(img,img,100,250);


	imshow("open", img);
	waitKey();
}
