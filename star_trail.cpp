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
#define TrailStyle 1
#define minutes 360
using namespace cv;
using namespace std;


Mat StarTrail(Point center, const Mat& img, Mat& result) {
	//Point center(img.cols / 2, img.rows / 2);
	Mat rot_mat;

	if (TrailStyle == 1) {
		rot_mat = getRotationMatrix2D(center, 0.25, 1.0);
	}
	else if (TrailStyle == 2) {
		rot_mat = getRotationMatrix2D(center, 0.8, 0.995);
	}
	else if (TrailStyle == 3) {
		rot_mat = getRotationMatrix2D(center, 0, 0.995);
	}

	Size img_size(img.cols, img.rows);
	//int times = minutes / ;
	Mat temp;
	threshold(img, img, 0, 255, THRESH_BINARY);

	warpAffine(img, temp, rot_mat, img_size, INTER_LINEAR);	//轉第一下

	// 使用BORDER_CONSTANT，並指定邊界顏色為黑色
	for (int i = 0; i < minutes; i++) {
		warpAffine(temp, temp, rot_mat, img_size, INTER_LINEAR);
		bitwise_or(temp, result, result);
	}


	GaussianBlur(result, result, Size(3, 3), 0.2, 0.2);
	cvtColor(result, result, COLOR_GRAY2BGR);
	cout << "Star Trail's type is: " << result.type() << endl;

	Mat mask;
	threshold(result, mask, 220, 255, THRESH_BINARY_INV);
	cout << "mask's type is: " << mask.type() << endl;

	imshow("Star Trail", result);

	return mask;
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

void ImageCombine(const Mat& img, const Mat& trail_mask, const Mat& trail, Mat& front) {
	cout << "我在ImageCombine" << endl;
	cout << "trail_mask " << trail_mask.type() << endl;
	cout << "img " << img.type() << endl;
	cout << "trail " << trail.type() << endl;
	cout << "front " << front.type() << endl;
	imshow("前景物件", front);


	bitwise_and(img, trail_mask, trail_mask);	//建立遮罩讓星軌更乾淨
	imshow("ROI", trail_mask);
	//waitKey();


	Mat floatImage = img.clone();/*
	floatImage.convertTo(floatImage, CV_32F);
	normalize(floatImage, floatImage, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);*/
	vector<cv::Mat> channels;
	split(floatImage, channels);

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

	// Set the clip limit (adjust as needed)
	clahe->setClipLimit(0.5);
	// Apply CLAHE to the input image
	cv::Mat outputImage;
	cv::Size gridSize(50, 50);  // You can change this to control the grid size
	clahe->setTilesGridSize(gridSize);


	// 對影像作histogram equalization
	for (int i = 0; i < channels.size(); ++i) {
		//cv::Mat channel8U = channels[i];
		//channels[i].convertTo(channel8U, CV_8U);
		//cout << channel8U.type() << endl;
		// 將通道轉換回 CV_32F
		//channel8U.convertTo(channels[i], CV_32F, 1.0 / 255.0);

		//cv::equalizeHist(channels[i], channels[i]);
		clahe->apply(channels[i], channels[i]);
	}

	// 合併通道
	merge(channels, floatImage);
	// 顯示均衡化後的圖像
	imshow("Original Float Image", floatImage);

	////將加強過的影像和front mask取出前景物件
	////normalize(floatImage, floatImage, 0.0, 1.0, NORM_MINMAX);
	//split(floatImage, channels);
	//Mat blueChannel, greenChannel, redChannel;
	//normalize(channels[0], blueChannel, 0, 255, cv::NORM_MINMAX);
	//channels[0].convertTo(blueChannel, CV_8U);
	//channels[1].convertTo(greenChannel, CV_8U);
	//channels[2].convertTo(redChannel, CV_8U);
	//normalize(blueChannel, blueChannel, 0, 255, cv::NORM_MINMAX);
	//normalize(greenChannel, greenChannel, 0, 255, cv::NORM_MINMAX);
	//normalize(redChannel, redChannel, 0, 255, cv::NORM_MINMAX);

	//imshow("CV_8U blueChannel Image", channels[0]);
	//cout << "channels[0] type = " << channels[0].type() << endl;
	//imshow("CV_8U greenChannel Image", greenChannel);
	//imshow("CV_8U redChannel Image", redChannel);


	//bitwise_and(blueChannel, front, blueChannel);
	//bitwise_and(greenChannel, front, greenChannel);
	//bitwise_and(redChannel, front, redChannel);


	//imshow("blueChannel Image", blueChannel);
	//imshow("greenChannel Image", greenChannel);
	//imshow("redChannel Image", redChannel);
	Mat floatMask;
	Mat frontlayer = Mat::zeros(img.size(), CV_8UC3);
	front.convertTo(floatMask, CV_8U);
	cvtColor(floatMask, floatMask, cv::COLOR_GRAY2BGR);
	cout << "floatImage type = " << floatImage.type() << endl;
	cout << "floatMask type = " << floatMask.type() << endl;
	bitwise_and(floatImage, floatMask, frontlayer);

	// 合併通道
	//Mat frontlayer = Mat::zeros(img.size(), CV_8UC3);
	//merge(channels, frontlayer);
	////merge(std::vector<cv::Mat>{blueChannel, greenChannel, redChannel}, frontlayer);
	//cout << frontlayer.type() << endl;

	//frontlayer.convertTo(frontlayer, CV_8U);
	//cout << frontlayer.type() << endl;
	frontlayer.convertTo(frontlayer, CV_8UC3);
	cout << frontlayer.type() << endl;
	imshow("frontlayer", frontlayer*255);
		


	Mat result = Mat::zeros(img.size(), CV_8UC3);
	add(trail, trail_mask, result);
	//addWeighted(trail, 1.0, trail_mask, 1.0, 0.0, result);
	imshow("result_startrail", result);

	/*-------------------------------------------------------------------------------------前景疊加不行----------*/
	//for (int i = 0; i < front.rows; ++i) {
	//	for (int j = 0; j < front.cols; ++j) {
	//		// 取得元素值
	//		int maskValue = front.at<int>(i, j);
	//		
	//		// 根據遮罩的像素值選擇 imgA 或 imgB 的像素值
	//		if (maskValue == 255) {
	//			result.at<cv::Vec3b>(i, j) = frontlayer.at<cv::Vec3b>(i, j);
	//		}
	//		else {
	//			result.at<cv::Vec3b>(i, j) = result.at<cv::Vec3b>(i, j);
	//		}
	//		// 在這裡進行你的操作，例如打印元素值
	//		//std::cout << "Element at (" << i << ", " << j << "): " << img << std::endl;
	//	}
	//}
	/*-------------------------------------------------------------------------------------前景疊加不行----------*/

	add(result, frontlayer, result);
	imshow("result_startrail_forntlayer", result);

	waitKey();
}

int main()
{
	double START = clock();
	string fileName = "./img./aurora_1.jpg";
	Mat img = imread(fileName);

	//resize(img, img, Size(img.cols * 0.6, img.rows * 0.6));
	resize(img, img, Size(500, 500));

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
	//cv::imshow("final front ", front);
	//cv::imshow("segmentedImage_result", segmentedImage);


	///*--------------------------------------------------------------*/
	star_result.convertTo(star_result, CV_8U);
	Mat trail;
	star_result.copyTo(trail);
	trail.create(Size(img.cols, img.rows), CV_8U);

	Mat trail_mask;
	trail_mask = StarTrail(center, star_result, trail);
	ImageCombine(org_img, trail_mask, trail, front);

	double END = clock();
	cout << "\n\nThe total time is: " << (END - START) / CLOCKS_PER_SEC << " sec." << endl;

	////imwrite("startrail.png", trail);

	/*--------------------------------------------------------------------*/
	waitKey();
	return 0;
}
