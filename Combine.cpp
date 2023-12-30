#include <opencv2/opencv.hpp>
#include <string>
#include <Vector>
#include <time.h>
#define PI 3.14159
#define flag 3
using namespace cv;
using namespace std;

void ImageCombine(const Mat& img, const Mat& trail_mask, const Mat& trail) {
	cout << trail_mask.type() << endl;
	cout << img.type() << endl;
	cout << trail.type() << endl;
	bitwise_and(img, trail_mask, trail_mask);	//建立遮罩讓星軌更乾淨
	//imshow("ROI", trail_mask);
	//waitKey();


	Mat result = Mat::zeros(img.size(), CV_8UC3);
	add(trail, trail_mask, result);
	imshow("result", result);
	waitKey();
}

Mat StarTrail(const Mat& img, Mat& result, double minutes) {
	Point center(img.cols / 4, img.rows / 2);
	Mat rot_mat;
	// 找最大星星
	if (flag == 1) {
		rot_mat = getRotationMatrix2D(center, 0.5, 1.0);
	}
	else if (flag == 2) {
		rot_mat = getRotationMatrix2D(center, 0.8, 0.995);
	}
	else if (flag == 3) {
		rot_mat = getRotationMatrix2D(center, 0, 0.998);
	}

	Size img_size(img.cols, img.rows);
	//int times = minutes / ;
	Mat temp;
	warpAffine(img, temp, rot_mat, img_size, INTER_LINEAR);	//轉第一下

	// 使用BORDER_CONSTANT，並指定邊界顏色為黑色
	for (int i = 0; i < 2 * minutes; i++) {
		warpAffine(temp, temp, rot_mat, img_size, INTER_LINEAR);
		bitwise_or(temp, result, result);
	}

	GaussianBlur(result, result, Size(3, 3), 0.2, 0.2);
	cvtColor(result, result, COLOR_GRAY2BGR);
	cout << "Star Trail's type is: " << result.type() << endl;

	Mat mask;
	threshold(result, mask, 220, 255,THRESH_BINARY_INV);
	cout << "mask's type is: " << mask.type() << endl;

	imshow("Star Trail", result);
	//imshow("Mask", mask);
	//waitKey(0);	
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

cv::Mat kmeans_seg(const cv::Mat org_img, int k) {
	Mat imageo = org_img.clone();
	cv::cvtColor(imageo, imageo, COLOR_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(imageo, channels);
	cv::Mat vChannel = channels[2];
	Mat Output_gamma2;
	gammaTransform(vChannel, Output_gamma2, 0.5);
	Mat image = Output_gamma2;
	int kernel_size = 15;
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

	int flags = 4 | (255 << 8);

	cv::Mat mask;
	mask.create(image.rows + 2, image.cols + 2, CV_8U);
	mask = cv::Scalar::all(0);

	cv::floodFill(image, mask, seedPoint, fillColor, nullptr, cv::Scalar(loDiff), cv::Scalar(upDiff), flags);

	return mask;
}

int main()
{
	double START = clock();
	string fileName = "./img./aurora_3.jpg";
	Mat img = imread(fileName);

	//resize(img, img, Size(img.cols, img.rows));
	resize(img, img, Size(700,700));

	Mat org_img = img;
	//cv::imshow("open", img);

	int blocksize = 10;
	int gridWidth = img.cols / blocksize;
	int gridHeight = img.rows / blocksize;


	cv::Mat star_mask;
	cv::Mat star_result = get_star(img, star_mask);
	//cv::imshow("starresult", star_result);
	//waitKey();


	cv::cvtColor(img, img, COLOR_BGR2GRAY);

	//Mat segmentedImage;
	//Mat segmentedImage = kmeans_seg(org_img, 4);
	//vector<Point>star_location = get_star_location(star_result);
	//cv::imshow("segmentedImage_result", segmentedImage);
	//for (int i = 0; i < star_location.size(); i++) {
	//	cout << star_location[i].x << " " << star_location[i].y << endl;
	//	Mat mymask = customFloodFill(segmentedImage, int(star_location[i].y), int(star_location[i].x), 0);
	//}
	//imshow("segmentedImage_result2", segmentedImage);
	//waitKey();

	/*--------------------------------------------------------------*/

	star_result.convertTo(star_result, CV_8U);
	Mat trail;
	star_result.copyTo(trail);
	trail.create(Size(img.cols, img.rows), CV_8U);
	
	Mat trail_mask;
	int miniute = 120;
	trail_mask = StarTrail(star_result, trail, miniute);
	ImageCombine(org_img, trail_mask, trail);
	double END = clock();
	cout << "\n\nThe total time is: " << (END - START)/ CLOCKS_PER_SEC << " sec." << endl;

	//psiralTrail(img, trail, 10);
	//imwrite("startrail.png", trail);

	/*--------------------------------------------------------------------*/
	return 0;
}
