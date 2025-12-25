/**
 * @file orb_cv.cpp
 * @brief 使用 OpenCV 自带的 ORB 特征提取与匹配示例程序
 *
 * 本程序演示如何：
 * 1. 读取两张图像；
 * 2. 使用 ORB 检测 Oriented FAST 关键点和计算 BRIEF 描述子；
 * 3. 使用暴力匹配（BruteForce-Hamming）对描述子进行匹配；
 * 4. 根据距离阈值筛选“好”匹配；
 * 5. 可视化所有匹配与筛选后的匹配。
 *
 * 依赖：OpenCV ≥ 4.0
 *
 * 编译（示例）：
 *   g++ orb_cv.cpp -o orb_cv `pkg-config --cflags --libs opencv4`
 *
 * 运行（示例）：
 *   ./orb_cv ../1.png ../2.png
 *
 * 结果：
 *   - 控制台输出特征提取与匹配耗时；
 *   - 弹出三个窗口：
 *     1. ORB features：第一张图的关键点；
 *     2. all matches：所有匹配；
 *     3. good matches：筛选后的匹配。
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// ./orb_cv ../dev_0_00X6_20251209_151131_281_rgb.png ../dev_4_00YA_20251209_151131_281_rgb.png
int main(int argc, char **argv){
	if (argc != 3){
		cout << "usage: feature_extraction img1 img2" << endl;
		return 1;
	}
	//-- 读取图像，根据命令行参数提供的图片路径读取图片
	Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);
	Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);
	assert(img_1.data != nullptr && img_2.data != nullptr);		// 确保图像1和图像2不为空

	//-- 初始化
	std::vector<KeyPoint> keypoints_1, keypoints_2;		// 初始化两个关键点集向量
	Mat descriptors_1, descriptors_2;						// 初始化两个描述子矩阵
	Ptr<FeatureDetector> detector = ORB::create();			// 创建 ORB 特征检测器
	Ptr<DescriptorExtractor> descriptor = ORB::create();	// 创建 ORB 描述子提取器
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");	// 创建基于 Hamming 距离的描述子匹配器		

	//-- 第一步:检测 Oriented FAST 角点位置
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();		// 开始计时
	detector->detect(img_1, keypoints_1);		// 检测图像1中的Oriented FAST 角点位置
	detector->detect(img_2, keypoints_2);		// 检测图像2中的Oriented FAST 角点位置

	//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();		// 结束计时
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);		// 计算角点检测和BRIEF描述子计算耗时
	cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

	Mat outimg1;
	// 在图像中绘制检测到的Oriented Fast角点，绘制好的图像导出为outimg1，Scalar::all(-1)表示特征点使用随机颜色绘制，采用默认绘制模式
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS表示绘制圆圈的大小对应特征点的尺度，圆圈内还有一条半径线指向特征点的方向（这就是 Oriented FAST 的“方向”可视化）
	// drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("ORB features", outimg1);

	//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
	vector<DMatch> matches;
	t1 = chrono::steady_clock::now();
	matcher->match(descriptors_1, descriptors_2, matches);
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

	//-- 第四步:匹配点对筛选
	// 计算最小距离和最大距离
	auto min_max = minmax_element(matches.begin(), matches.end(),
								  [](const DMatch &m1, const DMatch &m2)
								  { return m1.distance < m2.distance; });
	double min_dist = min_max.first->distance;
	double max_dist = min_max.second->distance;

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	// 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
	std::vector<DMatch> good_matches;
	for (int i = 0; i < descriptors_1.rows; i++){
		if (matches[i].distance <= max(2 * min_dist, 30.0)){
			good_matches.push_back(matches[i]);
		}
	}

	//-- 第五步:绘制匹配结果
	Mat img_match;
	Mat img_goodmatch;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
	imshow("all matches", img_match);
	imshow("good matches", img_goodmatch);
	waitKey(0);

	return 0;
}
