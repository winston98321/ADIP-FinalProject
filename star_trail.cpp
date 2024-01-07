#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <Vector>
#include<iostream>
#include <set>
#include <opencv2/core.hpp>

#include <time.h>


//#include<opencv2/gpu/gpu.hpp>

#define MY_PI 3.1415
#define TrailStyle 3
#define minutes 600
using namespace cv;
using namespace std;

void gammaTransform(const cv::Mat& input_image, cv::Mat& output_image, double gamma) {
	// Make sure the input image is not empty
	if (input_image.empty()) {
		std::cerr << "Error: Input image is empty." << std::endl;
		return;
	}

	// Create a copy of the input image to avoid modifying the original
	output_image = input_image.clone();

	// Normalize the pixel values to the range [0, 1]
	output_image.convertTo(output_image, CV_32F, 1.0 / 255.0);

	// Apply gamma transformation
	cv::pow(output_image, gamma, output_image);

	// Rescale the pixel values back to the range [0, 255]
	output_image *= 255.0;
	output_image.convertTo(output_image, CV_8U);
}

Mat StarTrail(Point center, const Mat& img, Mat& result) {
	//Point center(img.cols / 2, img.rows / 2);
	Mat rot_mat;
	Size img_size(img.cols, img.rows);

	//int times = minutes / ;
	Mat temp;
	threshold(img, img, 0, 255, THRESH_BINARY | THRESH_OTSU);

	cout << "temp's size" << temp.size() << endl;
	cout << "result's size" << result.size() << endl;


	// 使用BORDER_CONSTANT，並指定邊界顏色為黑色

	double scale = 0.99;
	//for (int i = 0; i < minutes; i++) {
	for (double radius = 0; radius < minutes / 60 * 15; radius += 1) {
		if (TrailStyle == 1) {
			rot_mat = getRotationMatrix2D(center, radius, 1.0);
		}
		else if (TrailStyle == 2) {
			rot_mat = getRotationMatrix2D(center, radius, scale);
		}
		else if (TrailStyle == 3) {
			rot_mat = getRotationMatrix2D(center, 0, scale);
		}
		warpAffine(img, temp, rot_mat, img_size);	//轉第一下
		//warpAffine(temp, temp, rot_mat, img_size);
		//threshold(temp, temp, 0, 255, THRESH_BINARY | THRESH_OTSU);
		//bitwise_or(temp, result, result);
		//add(temp, result, result);
		addWeighted(result, 1.0, temp, 0.8, 0.0, result);
		scale *= 0.99;
	}

	GaussianBlur(result, result, Size(3, 3), 0.5,0.5);
	cvtColor(result, result, COLOR_GRAY2BGR);
	cout << "Star Trail's type is: " << result.type() << endl;
	imshow("Star Trail", result);

	return result;
}


Mat get_star(const cv::Mat img, cv::Mat& star_mask) {
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


cv::Mat hsv_kmeans_seg(cv::Mat org_img, int k) {
	std::vector<cv::Mat> imgRGB, imgLab, imgHSV;
	cv::cvtColor(org_img, org_img, COLOR_BGR2Lab);
	cv::split(org_img, imgLab);

	cv::cvtColor(org_img, org_img, COLOR_Lab2RGB);
	cv::split(org_img, imgRGB);

	cv::cvtColor(org_img, org_img, COLOR_RGB2HSV);
	cv::split(org_img, imgHSV);
	cv::cvtColor(org_img, org_img, COLOR_HSV2BGR);

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

	// Set the clip limit (adjust as needed)
	clahe->setClipLimit(0.5);
	// Apply CLAHE to the input image
	cv::Mat outputImage;
	cv::Size gridSize(50, 50);  // You can change this to control the grid size
	clahe->setTilesGridSize(gridSize);

	// Apply CLAHE to the input image
	//cv::Mat claheImage;
	//clahe->apply(image, claheImage);
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
		//equalizeHist(imgLab[i], imgLab[i]);
		//equalizeHist(imgRGB[i], imgRGB[i]);

	}

	//for (int i = 0; i < 3; ++i) {
	//	blur(imgRGB[i], imgRGB[i], cv::Size(3,3));
	//}

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

Mat guidedFilter(const Mat& I, const Mat& p, int r, double eps) {
	Mat mean_I, mean_p, mean_Ip, mean_II, cov_Ip;

	// Mean and covariance calculations
	boxFilter(I, mean_I, -1, Size(r, r));
	boxFilter(p, mean_p, -1, Size(r, r));
	boxFilter(I.mul(p), mean_Ip, -1, Size(r, r));
	boxFilter(I.mul(I), mean_II, -1, Size(r, r));

	cov_Ip = mean_Ip - mean_I.mul(mean_p);

	// Var and mean calculations
	Mat var_I = mean_II - mean_I.mul(mean_I);
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);

	// Filter output calculation
	Mat mean_a, mean_b;
	boxFilter(a, mean_a, -1, Size(r, r));
	boxFilter(b, mean_b, -1, Size(r, r));

	return mean_a.mul(I) + mean_b;
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

cv::Mat customFloodFill(cv::Mat& image, int x, int y, cv::Scalar fillColor) {



	cv::Point seedPoint(x, y);
	int loDiff = 10;  // 低灰度差異閾值
	int upDiff = 10;  // 高灰度差異閾值

	int flags = 4 | (255 << 8);

	cv::Mat mask;
	mask.create(image.rows + 2, image.cols + 2, CV_8U);
	mask = cv::Scalar::all(0);

	cv::floodFill(image, mask, seedPoint, fillColor, nullptr, cv::Scalar(loDiff), cv::Scalar(upDiff), flags);

	return mask;
}

cv::Mat mySobel(cv::Mat image) {
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

	// Set the clip limit (adjust as needed)
	clahe->setClipLimit(0.5);
	// Apply CLAHE to the input image
	cv::Mat outputImage;
	cv::Size gridSize(20, 20);  // You can change this to control the grid size
	clahe->setTilesGridSize(gridSize);

	// Apply CLAHE to the input image
	for (int i = 0; i < 3; ++i) {
		cv::medianBlur(channels[i], channels[i], 9);  // Adjust the second parameter (kernel size) as needed
		blur(channels[i], channels[i], cv::Size(3, 3));
		channels[i] = guidedFilter(channels[i], channels[i], 3, 0.001);
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
		int max_row = 9999;  // 初始化最高?的行?

		// 找到?前列的最高?
		for (int row = 0; row < result.rows; row++) {
			if (static_cast<int>(result.at<uchar>(row, col)) >= 10) {
				max_row = min(row, max_row);
			}
		}

		// ?最高?下面的像素?置?255
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
	//Canny(star_result, result, 5, 30);
	std::vector<std::vector<cv::Point>> contours1;
	std::vector<cv::Vec4i> hierarchy1;
	Mat find_contour = result.clone();
	cv::Rect ROI2(left_up.x, left_up.y, abs(right_down.x - left_up.x), abs(right_down.y - left_up.y));		//ROI-->ROI2
	cv::Mat croppedImage = find_contour(ROI2);
	cv::findContours(croppedImage, contours1, hierarchy1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	double area = -1;
	int index = -1;

	for (int i = 0; i < contours1.size(); i++) {
		//cout << i << " " << area << " " << contours[i] << endl;
		if (cv::contourArea(contours1[i]) > area) {
			area = contourArea(contours1[i]);
			index = i;
		}
	}

	Point center(-1, -1);

	if (index >= 0) {

		Moments mu = moments(contours1[index]);
		center = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
		center.x += left_up.x;
		center.y += left_up.y;
	}
	//如果沒找到直接回傳圖片中最大星星位置
	else {
		cv::findContours(find_contour, contours1, hierarchy1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		double area = -1;
		int index = -1;

		for (int i = 0; i < contours1.size(); i++) {
			//cout << i << " " << area << " " << contours[i] << endl;
			if (cv::contourArea(contours1[i]) > area) {
				area = contourArea(contours1[i]);
				index = i;
			}
		}

		if (index >= 0) {
			Moments mu = moments(contours1[index]);
			center = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
			center.x += left_up.x;
			center.y += left_up.y;
		}
	}
	return center;
}

void histogramMapping(const cv::Mat& input_image, cv::Mat& output_image, int range_min, int range_max) {
	// Convert the input image to grayscale if it is a color image
	cv::Mat gray_image = input_image.clone();
	equalizeHist(input_image, gray_image);
	for (int i = 0; i < input_image.rows; i++) {
		for (int j = 0; j < input_image.cols; j++) {
			//output_image.at<uchar>(i, j) = range_min + (range_max - range_min) * (static_cast<double>(gray_image.at<uchar>(i, j)) / 255);
		}
	}
	output_image = gray_image.clone();

}

void removeContoursTouchingImageEdges(std::vector<std::vector<cv::Point>>& contours, cv::Size imageSize) {
	std::vector<std::vector<cv::Point>> validContours;

	for (const auto& contour : contours) {
		bool isTouchingEdge = false;
		bool isupedge = false;

		for (const auto& point : contour) {
			// Check if the point is near the image edge (e.g., within a certain distance)
			if (point.x < 5 || point.x > imageSize.width - 6 || point.y > imageSize.height / 2) {
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

void make_gif() {
	const char* command = "magick gif/*.jpg images.gif";
	//convert - resize 768x576 gif/*.jpg images.gif
	// 使用 system 函數執行命令
	int result = system(command);

	// 檢查執行結果
	if (result == 0) {
		// 成功
		std::cout << "GIF creation successful." << std::endl;
	}
	else {
		// 失敗
		std::cerr << "Error: Failed to create GIF." << std::endl;
	}
}
cv::Mat final_front(Mat org_img, Mat kmenas_seg) {
	Mat gray, th_re;
	cv::cvtColor(org_img, gray, COLOR_BGR2Lab);
	th_re = mySobel(gray);

	std::map<int, int> elementCount;
	std::map<int, int> frontCount;

	// 遍歷 th_re 的每一個元素
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

		std::cout << "元素 " << pair.first << " 出現了 " << pair.second << " 次。" << std::endl;
	}

	for (const auto& pair : frontCount) {
		std::cout << "元素 " << pair.first << " 出現了 " << pair.second << " 次。" << std::endl;
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
	//cv::imshow("kmenas_seg_bin", kmenas_seg);


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
	//cv::imshow("Contours without touching edges1", result);
	return result;
}


Mat clahe_front(const Mat& floatImage, const Mat& front) {
	vector<cv::Mat> channels, org_channels;
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

	// 對影像作histogram equalization
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
					channels[ch].at<uchar>(i, j) = org_channels[ch].at<uchar>(i, j);
				}
			}
		}
	}
	merge(channels, floatImage);

	return floatImage;
}

Mat ImageCombine(const Mat& img, const Mat& trail, const Mat& front) {

	Mat floatImage = img.clone();/*
	floatImage.convertTo(floatImage, CV_32F);
	normalize(floatImage, floatImage, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);*/


	cout << "front's type" << front.type() << endl;
	cout << "result's type" << img.type() << endl;

	// ��ܧ��Ťƫ᪺�Ϲ�
	cv::imshow("CLAHE Float Image", floatImage);
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
	cout << "trail_front_result's size" << trail_front_result.size() << endl;


	addWeighted(img, 1.0, trail_front_result, 0.5, 0, img);
	imshow("final Image", img);

	waitKey();
	return img;
}


int main()
{
	double START = clock();
	string fileName = "./img./aurora_5.jpg";
	Mat img = imread(fileName);

	/*if (img.cols > 1200 || img.rows > 1200) {
		resize(img, img, Size(img.cols * 0.3, img.rows * 0.3));
	}*/
	resize(img, img, Size(600,600));

	Mat org_img = img.clone();
	//cv::imshow("open", img);

	int blocksize = 10;
	int gridWidth = img.cols / blocksize;
	int gridHeight = img.rows / blocksize;


	cv::Mat star_mask;
	cv::Mat star_result = get_star(img, star_mask);
	cv::imshow("starresult", star_result);
	//waitKey();

	Point p1;
	p1.x = 0;
	p1.y = 0;
	Point p2;
	p2.x = 499;
	p2.y = 499;
	Point center = find_bigstar(star_result, p1, p2);
	//cout << "Big Star Location is: " << center << endl;


	/*---------------------------------------------------------------------------------------------以上為前景之外*/
	cv::cvtColor(img, img, COLOR_BGR2GRAY);

	//Mat segmentedImage;
	Mat segmentedImage = hsv_kmeans_seg(org_img, 8);
	Mat kmenas_seg = segmentedImage.clone();
	vector<Point>star_location = get_star_location(star_result);//所有星星位置，my median演算法中有用到
	//cv::cvtColor(org_img, org_img, COLOR_Lab2BGR);
	//最終取得前景的演算法
	Mat front;
	front = final_front(org_img, kmenas_seg);

	cv::imshow("取出前景 ", front);
	//cv::imshow("segmentedImage_result", segmentedImage);


	///*--------------------------------------------------------------*/

	star_result.convertTo(star_result, CV_8U);
	Mat trail;
	star_result.copyTo(trail);
	trail.setTo(0);
	trail.create(Size(img.cols, img.rows), CV_8U);
	//trail.zeros(trail.size(), trail.type());

	Mat enhance_img;
	enhance_img = clahe_front(org_img, front);

	Mat resultTrail, final_image;
	resultTrail = StarTrail(center, star_result, trail);
	//resultTrail = kaiStarTrail(center, star_result, trail);
	imshow("resultTrail", resultTrail);

	final_image = ImageCombine(enhance_img, trail, front);
	imshow("final_image", final_image);

	double END = clock();
	cout << "\n\nThe total time is: " << (END - START) / CLOCKS_PER_SEC << " sec." << endl;

	////imwrite("startrail.png", trail);

	/*--------------------------------------------------------------------*/
	waitKey();
	return 0;
}
