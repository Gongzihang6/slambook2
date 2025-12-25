#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp> // SIFT 头文件

using namespace std;
using namespace cv;

// ./SIFT_CV ../dev_0_00X6_20251209_151131_281_rgb.png ../dev_4_00YA_20251209_151131_281_rgb.png
// ./SIFT_CV ../g1.png ../g2.png
int main(int argc, char **argv) {
    // 1. 读取参数
    if (argc != 3) {
        cout << "Usage: ./SIFT_CV <img1> <img2>" << endl;
        return 1;
    }

    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);

    if (img1.empty() || img2.empty()) {
        cerr << "无法读取图像" << endl;
        return -1;
    }

    // ==========================================
    // 步骤 1: SIFT 特征检测与描述 (官方接口)
    // ==========================================
    // nfeatures: 0 表示不限制点数，保留所有特征点
    // nOctaveLayers: 3 (默认值)
    // contrastThreshold: 0.04 (默认值)
    // edgeThreshold: 10 (默认值)
    // sigma: 1.6 (默认值)
    Ptr<SIFT> sift = SIFT::create();

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;

    cout << "正在使用 OpenCV SIFT 提取特征..." << endl;
    double t1 = (double)getTickCount();
    
    // detectAndCompute 同时完成检测和计算描述子
    sift->detectAndCompute(img1, noArray(), kp1, des1);
    sift->detectAndCompute(img2, noArray(), kp2, des2);

    double t2 = (double)getTickCount();
    cout << "特征提取耗时: " << (t2 - t1) / getTickFrequency() * 1000 << " ms" << endl;
    cout << "图1 关键点: " << kp1.size() << ", 图2 关键点: " << kp2.size() << endl;

    // ==========================================
    // 步骤 2: 特征匹配 (k-NN + Ratio Test)
    // ==========================================
    if (des1.empty() || des2.empty()) return 0;

    // 使用 L2 范数 (SIFT 标准)
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    
    // k-NN 匹配: 对图1中的每个点，在图2中找 k=2 个最近邻
    // 这样做的目的是比较 "最近距离" 和 "次近距离"
    matcher.knnMatch(des1, des2, knn_matches, 2);

    // --- Lowe's Ratio Test (比率测试) ---
    // 原理: 如果最近邻距离 < 0.75 * 次近邻距离，说明匹配非常确信。
    // 如果两个距离差不多，说明特征点不独特，容易混淆，应该丢弃。
    const float ratio_thresh = 0.75f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    cout << "Ratio Test 后剩余匹配: " << good_matches.size() << endl;

    // ==========================================
    // 步骤 3: RANSAC 几何校验 (剔除局外点)
    // ==========================================
    vector<DMatch> final_matches;
    if (good_matches.size() >= 4) {
        vector<Point2f> src_pts;
        vector<Point2f> dst_pts;
        for (const auto &m : good_matches) {
            src_pts.push_back(kp1[m.queryIdx].pt);
            dst_pts.push_back(kp2[m.trainIdx].pt);
        }

        Mat mask;
        // 使用 RANSAC 计算单应性矩阵，允许误差 3.0 像素
        findHomography(src_pts, dst_pts, RANSAC, 3.0, mask);

        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<uchar>(i)) {
                final_matches.push_back(good_matches[i]);
            }
        }
        cout << "RANSAC 后最终匹配: " << final_matches.size() << endl;
    } else {
        final_matches = good_matches;
    }

    // ==========================================
    // 步骤 4: 可视化 (解决窗口过大问题)
    // ==========================================
    Mat img_matches;
    // flags=2 (NOT_DRAW_SINGLE_POINTS) 只画匹配线
    drawMatches(img1, kp1, img2, kp2, final_matches, img_matches, 
                Scalar::all(-1), Scalar::all(-1), vector<char>(), 
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    string win_name = "Official SIFT Matches";
    namedWindow(win_name, WINDOW_NORMAL); // 允许调整大小
    resizeWindow(win_name, 1200, 600);    // 设置初始大小
    imshow(win_name, img_matches);

    cout << "按 'q' 或 'ESC' 退出" << endl;
    while (true) {
        int key = waitKey(10);
        if (key == 'q' || key == 27) break;
        if (getWindowProperty(win_name, WND_PROP_VISIBLE) < 1) break;
    }

    return 0;
}
