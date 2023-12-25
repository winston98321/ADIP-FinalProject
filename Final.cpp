#include <opencv2/opencv.hpp>
#include <string>
#include <Vector>
#include<iostream>
using namespace cv;
using namespace std;
#define MY_PI 3.1415

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

Mat RGB2HSI(const Mat& rgb) {
	Mat hsi(rgb.rows, rgb.cols, rgb.type());
	float  H = 0, S = 0, I = 0;
	for (int i = 0; i < rgb.rows; i++)
		for (int j = 0; j < rgb.cols; j++) {
			float B = rgb.at<Vec3b>(i, j)[0] / 255.f,
				G = rgb.at<Vec3b>(i, j)[1] / 255.f,
				R = rgb.at<Vec3b>(i, j)[2] / 255.f;

			float num = (R - G + R - B) / 2,
				den = sqrt((R - G) * (R - G) + (R - B) * (G - B)),
				theta = acos(num / den);
			if (den == 0) H = 0; 
			else H = B <= G ? theta / (2 * MY_PI) : 1 - theta / (2 * MY_PI);

			float sum = B + G + R;
			if (sum == 0) S = 0;
			else S = 1 - 3 * min(min(B, G), R) / sum;

			I = sum / 3.0;

			hsi.at<Vec3b>(i, j)[0] = H * 255;
			hsi.at<Vec3b>(i, j)[1] = S * 255;
			hsi.at<Vec3b>(i, j)[2] = I * 255;
		}
	return hsi;
}
void showRGB_histogram(Mat img) {
	vector<Mat> bgr_planes;

	split(img, bgr_planes);


	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w = 512, hist_h = 400;
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));


	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++) {
		line(histImage, Point(hist_w / histSize * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(hist_w / histSize * i, hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(hist_w / histSize * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(hist_w / histSize * i, hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(hist_w / histSize * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(hist_w / histSize * i, hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("RGB Histogram", histImage);
	waitKey();
}
void showgrayscale_histogram(Mat img) {
	int histSize = 256;
	float range[] = { 0, 256 }; // Pixel value range
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	// Compute histogram
	Mat gray_hist;
	calcHist(&img, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Create histogram image
	int hist_w = 512, hist_h = 400;
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	// Normalize histogram data
	normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	// Draw histogram
	for (int i = 1; i < histSize; i++) {
		line(histImage, Point(hist_w / histSize * (i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1))),
			Point(hist_w / histSize * i, hist_h - cvRound(gray_hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}

	// Display histogram image
	imshow("Grayscale Histogram", histImage);
	waitKey(0);
}
void gammaTransform(cv::Mat& input_image, cv::Mat& output_image, double gamma) {
	// Make sure the input image is not empty
	if (input_image.empty()) {
		std::cerr << "Error: Input image is empty." << std::endl;
		return;
	}

	// Normalize the pixel values to the range [0, 1]
	input_image.convertTo(input_image, CV_32F, 1.0 / 255.0);

	// Apply gamma transformation
	cv::pow(input_image, gamma, output_image);

	// Rescale the pixel values back to the range [0, 255]
	output_image *= 255.0;
	output_image.convertTo(output_image, CV_8U);
}
int main()
{	
	string fileName = "./img./aurora_1.jpg";
	Mat img = imread(fileName);
	Mat org_img = img;
	resize(img, img, Size(500,500));

	imshow("open", img);
	waitKey();

	//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(12, 12));
	//morphologyEx(img, img, MORPH_OPEN, element);
	//Mat negativeImg = Mat::zeros(img.size(), img.type());
	//toNegative(img, negativeImg);
	//Mat LogImg = Mat::zeros(img.size(), img.type());

	////imgLog(img, LogImg);
	//histogram(img);
	int blocksize = 10;
	int gridWidth = img.cols / blocksize;
	int gridHeight = img.rows / blocksize;

	Scalar lowerBound(174, 159, 0);
	Scalar upperBound(255, 255, 255);
	Mat star_mask;
	inRange(img, lowerBound, upperBound, star_mask);
	
	Mat star_result;
	bitwise_and(img, img, star_result, star_mask);
	imshow("Star Mask", star_result);
	cvtColor(img, img, COLOR_BGR2GRAY);

	// 在10*10網格中找median number
	vector <int>median_list;
	for (int i = 0; i < blocksize; i++) {
		for(int j=0; j< blocksize;j++){
			vector<int>tmp;
			bool star_flag = false;
			for (int bl_i = i * gridHeight; bl_i < gridHeight * (i + 1); bl_i++) {
				for (int bl_j = j * gridHeight; bl_j < gridHeight * (j + 1); bl_j++) {
					
					tmp.push_back(static_cast<int>(img.at<uchar>(bl_i, bl_j)));
					if (star_mask.at<bool>(bl_i, bl_j) == 255) {;
						star_flag = true;
					}
				}
			}
			cout << " !!!!" << endl;
			nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
			int median;
			if (tmp.size() % 2 == 1) {
				median = tmp[tmp.size() / 2];
			}
			else {
				int middle1 = tmp[tmp.size() / 2 - 1];
				int middle2 = tmp[tmp.size() / 2];
				median = (middle1 + middle2) / 2;
			}
			
			if(star_flag)median_list.push_back(median);
			
		}
	}
	cout << "medianlist" << endl;
	for (int num : median_list) {

		std::cout << num << " ";
	}
	vector<int> test_list;
	nth_element(median_list.begin(), median_list.begin() + median_list.size() / 2, median_list.end());
	for (int i = 0; i < median_list.size(); i++) {
		if (test_list.size() == 0) {
			
			int median;
			if (median_list.size() % 2 == 1) {
				median = median_list[median_list.size() / 2];
			}
			else {
				int middle1 = median_list[median_list.size() / 2 - 1];
				int middle2 = median_list[median_list.size() / 2];
				median = (middle1 + middle2) / 2;
			}
			test_list.push_back(median);
		}
		else {
			bool test_flag = true;
			for (int j = 0; j < test_list.size(); j++) {
				if (median_list[i] < test_list[j] + 20 && median_list[i] > test_list[j] - 20) {
					test_flag = false;
				}
			}
			if (test_flag) {
				test_list.push_back(median_list[i]);
				cout << median_list[i] << endl;
			}
		}
	}
	
	cout << endl<<"testlist" << endl;
	for (int num : test_list) {

		std::cout << num << " ";
	}
		

	/*
	Mat hsi_img = RGB2HSI(img);
	Mat solo_img;
	extractChannel(hsi_img, solo_img, 0);*/

	// Apply Canny edge detection to the intensity component
	/*
	Mat edges;
	
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2GRAY);
	Mat Output_gamma;
	gammaTransform(hsv, Output_gamma,1.5);
	Canny(Output_gamma, edges, 100, 250);
	imshow("Canny Edges", edges);
	waitKey();

	cv::imshow("Gamma Transformed Image", Output_gamma);
	showRGB_histogram(img);
	showgrayscale_histogram(Output_gamma);*/
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
	
			int now = static_cast<int>(img.at<uchar>(i, j));
			
			for (int num : test_list) {
				if (now < num+25 && now > num-25) {
					uchar n = 255;
					
					img.at<uchar>(i, j) = n;
				}
			}
			
		}
	}imshow(" Threshold Image", img);
	/*
	medianBlur(img, img, 45);
	cvtColor(img, img, COLOR_BGR2GRAY);
	Vec3b a = img.at<Vec3b>(20, 20);
	Mat binaryImage;
	adaptiveThreshold(img, binaryImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 31, 1);

	// 顯示原始圖像和二值化圖像

	imshow("Adaptive Threshold Image", binaryImage);
	waitKey(0);



	
	
	cv::waitKey(0);*/
/*	for (int i = 1; i < 10; i++) {
		// 畫水平線
		line(img, Point(0, i * gridHeight), Point(imageWidth, i * gridHeight), Scalar(0, 255, 0), 2);

		// 畫垂直線
		line(img, Point(i * gridWidth, 0), Point(i * gridWidth, imageHeight), Scalar(0, 255, 0), 2);
	}*/

	// 顯示帶有網格的圖片
	imshow("Grid Image", img);
	waitKey(0);
	
	
	return 0;
}
