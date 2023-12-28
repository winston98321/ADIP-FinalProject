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
	cv::imshow("calcHist Demo", histImage);
	cv::waitKey();
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

	cv::imshow("RGB Histogram", histImage);
	cv::waitKey();
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
	cv::imshow("Grayscale Histogram", histImage);
	cv::waitKey(0);
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
cv::Mat get_star(const cv::Mat img, cv::Mat& star_mask) {
	cv::Scalar lowerBound(174, 159, 0);
	cv::Scalar upperBound(255, 255, 255);

	// Create a binary mask (star_mask) based on the color range
	cv::inRange(img, lowerBound, upperBound, star_mask);

	// Apply the binary mask to the original image
	cv::Mat star_result;
	cv::bitwise_and(img, img, star_result, star_mask);
	cv::cvtColor(star_result, star_result, COLOR_BGR2GRAY);

	// cv::imshow("Star Mask", star_result);
	
	// Find contours in the image and cancel too big contour set as 200
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	Mat star_process = star_result.clone();
	cv::dilate(star_process, star_process, cv::Mat());
	cv::dilate(star_process, star_process, cv::Mat());
	cv::dilate(star_process, star_process, cv::Mat());
	cv::dilate(star_process, star_process, cv::Mat());

	cv::findContours(star_process, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Draw contours on the original image
	cv::Mat contourImage = star_result.clone();
	cv::Mat mask = cv::Mat::ones(star_result.size(), CV_8UC1) * 255;
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
cv::Mat kmeans_seg(const cv::Mat org_img, int k) {
	Mat imageo = org_img.clone();
	cv::cvtColor(imageo, imageo, COLOR_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(imageo, channels);
	cv::Mat vChannel = channels[2];
	Mat Output_gamma2;
	gammaTransform(vChannel, Output_gamma2, 0.5);
	Mat image = Output_gamma2;
	int kernel_size =15;
	medianBlur(image, image, kernel_size);

	Mat reshapedImage = image.reshape(1, image.rows * image.cols);
	reshapedImage.convertTo(reshapedImage, CV_32F);

	// Set the number of clusters (k)
	

	// Set the criteria for k-means algorithm
	TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.2);
	Mat labels, centers;

	// Apply k-means algorithm
	cv::kmeans(reshapedImage, k, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

	// Convert centers to 8-bit for visualization
	centers.convertTo(centers, CV_8U);

	// Create a segmented image using cluster centers
	Mat segmentedImage(image.size(), CV_8U);

	for (int i = 0; i < reshapedImage.rows; ++i) {
		int clusterIdx = labels.at<int>(i);
		segmentedImage.at<uchar>(i / image.cols, i % image.cols) = centers.at<uchar>(clusterIdx);
	}

	// Display the original and segmented images
	cv::imshow("kmeans Image", segmentedImage);
	return segmentedImage;
}
void fillContourRegion(cv::Mat img, cv::Mat star_img) {
	Mat kmean_process = img.clone();
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	Mat edges;

	medianBlur(kmean_process, kmean_process, 9);

	Canny(kmean_process, edges, 30, 100);
	
	//cv::Mat openedImage;
	
	
	for (int i = 0; i < 500; i++) {
		for (int j = 0; j < 500; j++) {
			if (int(edges.at<uchar>(i, j)) > 128) {
				edges.at<uchar>(i, j) = 255;
			}
			else edges.at<uchar>(i, j) = 0;
		}
	}
	for (int i = 0; i < 500; i++) {
		edges.at<uchar>(i, 0) = 255;
		edges.at<uchar>(i, 499) = 255;
		edges.at<uchar>(0, i) = 255;
		edges.at<uchar>(499, i) = 255;
	}
	imshow("knn_Canny Edges1", edges);
	cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Draw contours on the original image
	cv::Mat contourImage = img.clone();
	cv::Mat mask = cv::Mat::ones(img.size(), CV_8UC1) * 255;
	//cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 255, 0), 2);
	cout << contours.size() <<"here"<< endl;
	Mat answer;
	for (int i = 0; i < star_img.cols; i++) {
		for (int j = 0; j < star_img.rows; j++) {
			if (int(star_img.at<uchar>(j, i)) != 0) {
				cout << i << " " << j << endl;
				for (int ii = 0; ii < contours.size(); ii++) {
					cv::Point testPoint(j, i);
					
					double distance = cv::pointPolygonTest(contours[ii], testPoint, false);
					cout << distance << endl;
					drawContours(mask, contours, int(ii), Scalar(0), -1, 8);//dst,contour,number of contour,color,fill/size of line/
					if (distance > 0) {
						mask.at<uchar>(i, j) = 0;
						//drawContours(mask, contours, int(ii), Scalar(0), -1, 8);//dst,contour,number of contour,color,fill/size of line/
							//cv::fillPoly(mask,  contours, cv::Scalar(0), 8, 0);
					}
				}
			}
			
			star_img.copyTo(star_img, mask);
			img.copyTo(answer, mask);
		}
	}
	cv::imshow("knn_mytest", mask);
}
//#https://www.educative.io/answers/flood-fill-algorithm-in-cpp
void myfloodFill(cv::Mat& image, int x, int y, int currColor, int newColor)
{
	// Base cases
	if (x < 0 || x >= image.cols || y < 0 || y >= image.rows)
		return;
	if (int(image.at<uchar>(x, y)) != currColor)
		return;
	if (int(image.at<uchar>(x, y)) == newColor)
		return;
	
	// Replace the color at cell (x, y)
	image.at<uchar>(x, y) = newColor;

	// Recursively call for north, east, south, and west
	myfloodFill(image, x + 1, y, currColor, newColor);
	myfloodFill(image, x - 1, y, currColor, newColor);
	myfloodFill(image, x, y + 1, currColor, newColor);
	myfloodFill(image, x, y - 1, currColor, newColor);
}

// It mainly finds the previous color on (x, y) and
// calls floodFill()
void findColor(cv::Mat& image, int x, int y, int newColor)
{
	int currColor = int(image.at<uchar>(y, x));
	myfloodFill(image, x, y, currColor, newColor);
}
vector<Point> get_star_location(Mat image) {
	vector<Point>ans;
	for (int i = 0; i < image.cols; i++) {
		for (int j = 0; j < image.rows; j++) {
			Point p;
			if (int(image.at<uchar>(i, j)) != 0) {
				p.x = i; p.y = j;
				ans.push_back(p);
			}
		}
	}
	return ans;
}
cv::Mat customFloodFill(cv::Mat image, int x, int y, cv::Scalar fillColor) {
	cv::Point seedPoint(x, y);
	int loDiff = 20;  // 低灰度差異閾值
	int upDiff = 20;  // 高灰度差異閾值

	int flags = 4  | (255 << 8);

	cv::Mat mask;
	mask.create(image.rows + 2, image.cols + 2, CV_8U);
	mask = cv::Scalar::all(0);

	cv::floodFill(image, mask, seedPoint, fillColor, nullptr, cv::Scalar(loDiff), cv::Scalar(upDiff), flags);

	return mask;
}
int main()
{
	string fileName = "./img./foreground_1.jpg";
	Mat img = imread(fileName);

	resize(img, img, Size(500, 500));
	Mat org_img = img;
	cv::imshow("open", img);


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


	cv::Mat star_mask;
	cv::Mat star_result = get_star(img, star_mask);
	cv::imshow("starresult",star_result);

	
	cv::cvtColor(img, img, COLOR_BGR2GRAY);

	//Mat segmentedImage;
	Mat segmentedImage = kmeans_seg(org_img,4);
	vector<Point>star_location = get_star_location(star_result);
	cv::imshow("segmentedImage_result", segmentedImage);
	for (int i = 0; i < star_location.size(); i++) {
		cout << star_location[i].x << " " << star_location[i].y << endl;
		Mat mymask = customFloodFill(segmentedImage, int(star_location[i].y), int(star_location[i].x), 0);
	}

	cv::imshow("segmentedImage_result2", segmentedImage);
	//
	//
	// 
	// 
	// 這下面目前是垃圾可以不用看 :(((
	// 
	// 
	// 
	// 
	// 
	//fillContourRegion(segmentedImage, star_result);
	

	// 在10*10網格中找median number
	vector <int>median_list;
	for (int i = 0; i < blocksize; i++) {
		for (int j = 0; j < blocksize; j++) {
			vector<int>tmp;
			bool star_flag = false;
			for (int bl_i = i * gridHeight; bl_i < gridHeight * (i + 1); bl_i++) {
				for (int bl_j = j * gridHeight; bl_j < gridHeight * (j + 1); bl_j++) {

					tmp.push_back(static_cast<int>(segmentedImage.at<uchar>(bl_i, bl_j)));
					if (star_mask.at<bool>(bl_i, bl_j) == 255) {
						;
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

			if (star_flag)median_list.push_back(median);

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

	cout << endl << "testlist" << endl;
	for (int num : test_list) {

		std::cout << num << " ";
	}


	/*
	Mat hsi_img = RGB2HSI(img);
	Mat solo_img;
	extractChannel(hsi_img, solo_img, 0);*/

	// Apply Canny edge detection to the intensity component

	Mat edges;

	Mat hsv= segmentedImage;
	//cvtColor(org_img, hsv, COLOR_BGR2GRAY);
	GaussianBlur(hsv, hsv, cv::Size(5, 5), 1.5, 1.5);
	Mat Output_gamma;
	//gammaTransform(hsv, Output_gamma,1.5);
	Canny(hsv, edges, 5, 30);
	imshow("Canny Edges", edges);


	//imshow("Gamma Transformed Image", Output_gamma);
	//showRGB_histogram(org_img);
	//showgrayscale_histogram(Output_gamma);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			int now = static_cast<int>(segmentedImage.at<uchar>(i, j));

			for (int num : test_list) {
				if (now < num + 10 && now > num - 10) {
					uchar n = 0;

					segmentedImage.at<uchar>(i, j) = n;
				}
				else {
					uchar n = 255;
					segmentedImage.at<uchar>(i, j) = n;
				}
			}

		}
	}imshow(" Threshold Image", segmentedImage);
	cv::Mat dilatedEdges;
	cv::dilate(img, dilatedEdges, cv::Mat());
	imshow(" dilatedEdges Image", dilatedEdges);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

	// Apply erosion
	cv::Mat erodedImage2;
	cv::erode(dilatedEdges, erodedImage2, kernel);
	cv::imshow(" erodedImage Image", erodedImage2);
	cv::dilate(erodedImage2, dilatedEdges, cv::Mat());
	cv::imshow(" dilatedEdges Image", dilatedEdges);
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
