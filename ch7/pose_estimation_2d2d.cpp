#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// #include "extra.h" // use this if in OpenCV2

using namespace std;
using namespace cv;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

// 寻找两张图像中的匹配点
void find_feature_matches(
	const Mat &img_1, const Mat &img_2,
	std::vector<KeyPoint> &keypoints_1,
	std::vector<KeyPoint> &keypoints_2,
	std::vector<DMatch> &matches);

// 估计两张图像之间相机的运动
void pose_estimation_2d2d(
	std::vector<KeyPoint> keypoints_1,
	std::vector<KeyPoint> keypoints_2,
	std::vector<DMatch> matches,
	Mat &R, Mat &t);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// ./pose_estimation_2d2d ../1.png ../2.png
// ./pose_estimation_2d2d ../dev_0_00X6_20251209_151131_281_rgb.png ../dev_4_00YA_20251209_151131_281_rgb.png
int main(int argc, char **argv){
	if (argc != 3){
		cout << "usage: pose_estimation_2d2d img1 img2" << endl;
		return 1;
	}
	//-- 读取图像
	/**
	 * cv::IMREAD_COLOR 表示强制以彩色方式读取，返回3通道8-bit图像，通常是CV_8UC3，通道顺序是BGR
	 * 如果原图有alpha通道，会丢弃alpha，只保留BGR
	 * 
	 * cv::IMREAD_GRAYSCALE 表示以灰度图读取（单通道），通常得到 CV_8UC1
	 * 
	 * cv::IMREAD_UNCHANGED 按原样读取：保留通道数（含 alpha）、保留位深（例如 16-bit PNG/TIFF），不做强制转换。
	 * 
	 * cv::IMREAD_COLOR_RGB 强制 3 通道 RGB（而不是默认 BGR）。
	 * 
	 * 与“位深/颜色”有关的辅助 flags（可组合）
	 * cv::IMREAD_ANYDEPTH 若图像是 16-bit/32-bit，就按原位深读进来（例如 CV_16U / CV_32F），否则还是 8-bit。
	 * cv::IMREAD_ANYCOLOR 尽量按图像原本的颜色格式读（有时用于让解码器不要强制变灰/变 BGR）。
	 */
	Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);
	Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);
	assert(img_1.data && img_2.data && "Can not load images!");

	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "一共找到了" << matches.size() << "组匹配点" << endl;

	//-- 估计两张图像间运动
	Mat R, t;
	pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

	//-- 验证E=t^R*scale
	Mat t_x =
		(Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
		 t.at<double>(2, 0), 0, -t.at<double>(0, 0),
		 -t.at<double>(1, 0), t.at<double>(0, 0), 0);

	cout << "t^R=" << endl
		 << t_x * R << endl;

	//-- 验证对极约束
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	int count = 0;
	cout << "----- 验证开始: 汉明距离 vs 对极约束误差 -----" << endl;
	for (DMatch m : matches){
		Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
		Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
		Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
		Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

		// 计算对极约束 x2^T * E * x1
		Mat d = y2.t() * t_x * R * y1;

		// 获取标量值（因为d是一个1x1的矩阵）
        double epipolar_error = d.at<double>(0, 0);

        // 输出 汉明距离 和 对极约束误差
        // m.distance 就是两个描述子的汉明距离
        cout << "Match " << count++ << ": "
             << "Hamming Dist = " << m.distance << "\t"
             << "Epipolar Error = " << epipolar_error << endl;
	}
	return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
						  std::vector<KeyPoint> &keypoints_1,
						  std::vector<KeyPoint> &keypoints_2,
						  std::vector<DMatch> &matches){
	//-- 初始化
	Mat descriptors_1, descriptors_2;
	// used in OpenCV3
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	// use this if you are in OpenCV2
	// Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
	// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	//-- 第一步:检测 Oriented FAST 角点位置
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
	vector<DMatch> match;
	// BFMatcher matcher ( NORM_HAMMING );
	matcher->match(descriptors_1, descriptors_2, match);

	//-- 第四步:匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	// 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < descriptors_1.rows; i++){
		double dist = match[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	// 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
	for (int i = 0; i < descriptors_1.rows; i++){
		if (match[i].distance <= max(2 * min_dist, 30.0)){
			matches.push_back(match[i]);
		}
	}
}

// 将图像上的像素坐标（Pixel Coordinates）转换为相机归一化平面坐标（Normalized Camera Coordinates）
Point2d pixel2cam(const Point2d &p, const Mat &K){
	return Point2d(
		(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
		(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
						  std::vector<KeyPoint> keypoints_2,
						  std::vector<DMatch> matches,
						  Mat &R, Mat &t){
	// 相机内参,TUM Freiburg2
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

	//-- 把匹配点转换为vector<Point2f>的形式
	vector<Point2f> points1;
	vector<Point2f> points2;

	/**
	 * queryIdx (查询索引)：对应 match 函数的第一个参数（这里是 img_1 的描述子）。
	 * trainIdx (训练索引)：对应 match 函数的第二个参数（这里是 img_2 的描述子）。
	 */
	for (int i = 0; i < (int)matches.size(); i++){
		// --- 第一句：提取图1中的对应点坐标 ---
		// matches[i].queryIdx ：当前匹配对在“查询图”（也就是图1）中的索引
		// keypoints_1[...]    ：拿到图1中对应的那个 KeyPoint 对象
		// .pt                 ：只取坐标 (x, y)，扔掉方向、大小等其他信息
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		// --- 第二句：提取图2中的对应点坐标 ---
		// matches[i].trainIdx ：当前匹配对在“训练图”（也就是图2）中的索引
		// keypoints_2[...]    ：拿到图2中对应的那个 KeyPoint 对象
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	//-- 计算基础矩阵
	Mat fundamental_matrix;
	/**
	 * Mat cv::findFundamentalMat(
		InputArray points1,  // 图1的坐标点集 (vector<Point2f>)
		InputArray points2,  // 图2的坐标点集 (vector<Point2f>)
		int method = FM_RANSAC, // 求解方法（重点！）
		double ransacReprojThreshold = 3.0, // RANSAC 重投影误差阈值
		double confidence = 0.99, // 置信度
		int maxIters = 2000 // 最大迭代次数
);
	 * cv::FM_8POINT	八点法	经典的线性解法，速度快。	仅适用于匹配点非常精确、无误匹配的情况（如教科书演示）。
	 * cv::FM_RANSAC	RANSAC	随机采样一致性。从点集中随机选8个点算模型，看剩余点有多少符合该模型。	工程实战首选。能剔除大量误匹配（Outliers）。
	 * cv::FM_LMEDS	LMedS	最小中值平方法。	不需要设定阈值，但在误匹配率 > 50% 时失效。
	 */
	fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
	cout << "fundamental_matrix is " << endl
		 << fundamental_matrix << endl;

	//-- 计算本质矩阵
	Point2d principal_point(325.1, 249.7); // 相机光心, TUM dataset标定值，就是内参里面的c_x，c_y
	double focal_length = 521;			   // 相机焦距, TUM dataset标定值
	Mat essential_matrix;

	// 需要传入相机内参 focal_length, principal_point，将像素坐标转换为相机归一化平面坐标，根据x_2^T E x_1 计算本质矩阵
	essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
	cout << "essential_matrix is " << endl
		 << essential_matrix << endl;

	//-- 计算单应矩阵
	//-- 但是本例中场景不是平面，单应矩阵意义不大
	Mat homography_matrix;
	homography_matrix = findHomography(points1, points2, RANSAC, 3);
	cout << "homography_matrix is " << endl
		 << homography_matrix << endl;

	//-- 从本质矩阵中恢复旋转和平移信息.
	// 此函数仅在Opencv3中提供
	recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
	cout << "R is " << endl
		 << R << endl;
	cout << "t is " << endl
		 << t << endl;
}
