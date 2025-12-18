/*
 * 手动实现 SIFT 关键点检测与优化匹配 (命令行参数版)
 * * 使用方法:
 * Linux:   ./SIFT_self path/to/image1.png path/to/image2.png
 * Windows: SIFT_self.exe path/to/image1.png path/to/image2.png
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ================= SIFT 参数配置 =================
const int NUM_OCTAVES = 8;
const int SCALES_PER_OCTAVE = 3;
const double SIGMA_0 = 1.6;
const double CONTRAST_THRESH = 0.04;
const double EDGE_THRESH = 10.0;

// 自定义关键点结构
struct MyKeyPoint {
    Point2f pt;
    int octave;
    int scale_idx;
    float size;
};

class MySift {
public:
    void detect(const Mat &img, vector<KeyPoint> &keypoints) {
        vector<vector<Mat>> gaussian_pyramid;
        buildGaussianPyramid(img, gaussian_pyramid);

        vector<vector<Mat>> dog_pyramid;
        buildDoGPyramid(gaussian_pyramid, dog_pyramid);

        vector<MyKeyPoint> my_kps;
        findScaleSpaceExtrema(dog_pyramid, my_kps);

        keypoints.clear();
        for (auto &kp : my_kps) {
            float scale_factor = pow(2.0, kp.octave);
            KeyPoint cv_kp;
            cv_kp.pt = kp.pt * scale_factor;
            cv_kp.size = kp.size * scale_factor;
            keypoints.push_back(cv_kp);
        }
    }

    // 简易描述子 (4x4网格梯度)
    void compute(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors) {
        int d_width = 4; 
        int d_bins = 8;
        int desc_size = d_width * d_width * d_bins; 

        descriptors = Mat::zeros(keypoints.size(), desc_size, CV_32F);

        for (size_t i = 0; i < keypoints.size(); i++) {
            Point pt(keypoints[i].pt);
            int radius = 8;
            
            if (pt.x < radius || pt.y < radius || pt.x >= img.cols - radius || pt.y >= img.rows - radius)
                continue;

            int bin_idx = 0;
            for (int r = -radius; r < radius; r += radius/2) {
                for (int c = -radius; c < radius; c += radius/2) {
                    float dx = img.at<uchar>(pt.y + r, pt.x + c + 1) - img.at<uchar>(pt.y + r, pt.x + c - 1);
                    float dy = img.at<uchar>(pt.y + r + 1, pt.x + c) - img.at<uchar>(pt.y + r - 1, pt.x + c);
                    float mag = sqrt(dx*dx + dy*dy);
                    float ang = fastAtan2(dy, dx); 

                    int angle_bin = int(ang / 45.0) % 8;
                    descriptors.at<float>(i, bin_idx * 8 + angle_bin) += mag;
                    bin_idx++;
                    if(bin_idx >= 16) break;
                }
                if(bin_idx >= 16) break;
            }
            
            Mat row = descriptors.row(i);
            normalize(row, row);
            threshold(row, row, 0.2, 0.2, THRESH_TRUNC);
            normalize(row, row);
        }
    }

private:
    void buildGaussianPyramid(const Mat &img, vector<vector<Mat>> &pyramid) {
        Mat gray;
        if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
        else gray = img.clone();
        gray.convertTo(gray, CV_32F, 1.0/255.0);

        double k = pow(2.0, 1.0 / SCALES_PER_OCTAVE);

        for (int o = 0; o < NUM_OCTAVES; ++o) {
            vector<Mat> octave_imgs;
            Mat base;
            if (o == 0) base = gray.clone();
            else resize(pyramid[o - 1][SCALES_PER_OCTAVE], base, Size(), 0.5, 0.5, INTER_NEAREST);
            
            octave_imgs.push_back(base);
            for (int s = 1; s < SCALES_PER_OCTAVE + 3; ++s) {
                double sigma = SIGMA_0 * pow(k, s);
                Mat blurred;
                GaussianBlur(octave_imgs.back(), blurred, Size(0, 0), sigma);
                octave_imgs.push_back(blurred);
            }
            pyramid.push_back(octave_imgs);
        }
    }

    void buildDoGPyramid(const vector<vector<Mat>> &g_pyr, vector<vector<Mat>> &dog_pyr) {
        for (const auto &octave : g_pyr) {
            vector<Mat> dog_octave;
            for (size_t i = 0; i < octave.size() - 1; ++i) {
                Mat dog;
                subtract(octave[i + 1], octave[i], dog);
                dog_octave.push_back(dog);
            }
            dog_pyr.push_back(dog_octave);
        }
    }

    void findScaleSpaceExtrema(const vector<vector<Mat>> &dog_pyr, vector<MyKeyPoint> &kps) {
        double k = pow(2.0, 1.0 / SCALES_PER_OCTAVE);
        
        for (int o = 0; o < NUM_OCTAVES; ++o) {
            for (int s = 1; s <= SCALES_PER_OCTAVE; ++s) {
                Mat img_prev = dog_pyr[o][s - 1];
                Mat img_curr = dog_pyr[o][s];
                Mat img_next = dog_pyr[o][s + 1];

                for (int r = 1; r < img_curr.rows - 1; ++r) {
                    for (int c = 1; c < img_curr.cols - 1; ++c) {
                        float val = img_curr.at<float>(r, c);
                        if (std::abs(val) < CONTRAST_THRESH) continue;

                        if (isExtremum(val, img_curr, img_prev, img_next, r, c)) {
                            float dxx = img_curr.at<float>(r, c+1) + img_curr.at<float>(r, c-1) - 2*val;
                            float dyy = img_curr.at<float>(r+1, c) + img_curr.at<float>(r-1, c) - 2*val;
                            float dxy = 0.25 * (img_curr.at<float>(r+1, c+1) - img_curr.at<float>(r+1, c-1) -
                                                img_curr.at<float>(r-1, c+1) + img_curr.at<float>(r-1, c-1));
                            
                            float trH = dxx + dyy;
                            float detH = dxx * dyy - dxy * dxy;

                            if (detH > 0 && (trH * trH / detH) < ((EDGE_THRESH + 1) * (EDGE_THRESH + 1) / EDGE_THRESH)) {
                                MyKeyPoint kp;
                                kp.pt = Point2f(c, r);
                                kp.octave = o;
                                kp.scale_idx = s;
                                kp.size = SIGMA_0 * pow(k, s);
                                kps.push_back(kp);
                            }
                        }
                    }
                }
            }
        }
    }

    bool isExtremum(float val, const Mat &curr, const Mat &prev, const Mat &next, int r, int c) {
        bool is_max = val > 0;
        bool is_min = val < 0;
        if (!is_max && !is_min) return false;

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                float v_prev = prev.at<float>(r + i, c + j);
                float v_curr = curr.at<float>(r + i, c + j);
                float v_next = next.at<float>(r + i, c + j);

                if (is_max) {
                    if (v_prev >= val || v_next >= val) return false;
                    if ((i != 0 || j != 0) && v_curr >= val) return false;
                } else if (is_min) {
                    if (v_prev <= val || v_next <= val) return false;
                    if ((i != 0 || j != 0) && v_curr <= val) return false;
                }
            }
        }
        return true;
    }
};


// cmake .. && make && ./SIFT_self ../g1.png ../g2.png
int main(int argc, char **argv) {
    // ---------------------------------------------------
    // 1. 命令行参数校验 
    // ---------------------------------------------------
    if (argc != 3) {
        cout << "Usage: SIFT_self <path_to_image1> <path_to_image2>" << endl;
        return 1;
    }
    // ./SIFT_self ../1.png ../2.png
    // 获取路径字符串，方便错误打印
    string img1_path = argv[1];
    string img2_path = argv[2];

    Mat img1 = imread(img1_path);
    Mat img2 = imread(img2_path);

    if (img1.empty() || img2.empty()) {
        cerr << "错误: 无法读取图像！" << endl;
        cerr << "路径1: " << img1_path << endl;
        cerr << "路径2: " << img2_path << endl;
        return -1;
    }

    // ---------------------------------------------------
    // 2. SIFT 检测与描述
    // ---------------------------------------------------
    MySift sift_detector;
    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;

    cout << "1. 正在检测关键点..." << endl;
    sift_detector.detect(img1, kp1);
    sift_detector.compute(img1, kp1, des1);

    sift_detector.detect(img2, kp2);
    sift_detector.compute(img2, kp2, des2);

    cout << "   图1 关键点数: " << kp1.size() << ", 图2 关键点数: " << kp2.size() << endl;

    // ---------------------------------------------------
    // 3. 匹配与筛选 (仿 ORB 策略)
    // ---------------------------------------------------
    if (des1.empty() || des2.empty()) {
        cerr << "警告: 未检测到足够的描述子，无法匹配。" << endl;
        return 0;
    }

    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(des1, des2, matches);
    cout << "2. 原始匹配点数: " << matches.size() << endl;

    // 计算最小距离
    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < matches.size(); i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    cout << "   最小距离: " << min_dist << ", 最大距离: " << max_dist << endl;

    // 筛选: 距离 < max(2 * min_dist, 0.3)
    vector<DMatch> good_matches;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= min(2 * min_dist, 0.8)) {
            good_matches.push_back(matches[i]);
        }
    }
    cout << "3. 优化后匹配点数: " << good_matches.size() << endl;

    // ---------------------------------------------------
    // 4. 绘图与防闪退显示
    // ---------------------------------------------------
    // Mat img_match;
    // drawMatches(img1, kp1, img2, kp2, good_matches, img_match);
    
    // ---------------------------------------------------
    // 4. 自定义绘制匹配结果 (解决线太细的问题)
    // ---------------------------------------------------
    
    // 1. 创建一张大图，能容纳两张图并排
    int height = max(img1.rows, img2.rows);
    int width = img1.cols + img2.cols;
    Mat img_match = Mat::zeros(height, width, CV_8UC3); // 彩色图

    // 2. 拷贝两张原图到大图上
    // 左边放 img1
    Mat left_roi = img_match(Rect(0, 0, img1.cols, img1.rows));
    if(img1.channels() == 1) cvtColor(img1, left_roi, COLOR_GRAY2BGR);
    else img1.copyTo(left_roi);

    // 右边放 img2
    Mat right_roi = img_match(Rect(img1.cols, 0, img2.cols, img2.rows));
    if(img2.channels() == 1) cvtColor(img2, right_roi, COLOR_GRAY2BGR);
    else img2.copyTo(right_roi);

    // 3. 绘制匹配线和点
    Scalar line_color(0, 255, 0); // 绿色连线
    Scalar pt_color(0, 0, 255);   // 红色关键点
    int line_thickness = 4;       // 线宽 (这里设为2，可以改更大)
    int pt_radius = 6;            // 点的半径

    for (const auto& m : good_matches) {
        // img1 的关键点坐标
        Point2f p1 = kp1[m.queryIdx].pt;
        // img2 的关键点坐标 (注意要加上 img1 的宽度偏移)
        Point2f p2 = kp2[m.trainIdx].pt + Point2f(img1.cols, 0);

        // 画点
        circle(img_match, p1, pt_radius, pt_color, -1); // -1表示实心
        circle(img_match, p2, pt_radius, pt_color, -1);

        // 画线
        line(img_match, p1, p2, line_color, line_thickness, LINE_AA);
    }

    // 4. 显示
    string win_name = "Optimized Matches";
    namedWindow(win_name, WINDOW_NORMAL); 
    resizeWindow(win_name, 1200, 800); 
    imshow(win_name, img_match);

    cout << "按 'q' 或 'ESC' 键退出程序..." << endl;

    // 循环监听按键，解决 Shift 截图自动退出的问题
    while (true) {
        int key = waitKey(10); // 等待 10ms
        if (key == 'q' || key == 27) { // 'q' 或 ESC
            break; 
        }
        // 检测到窗口被关闭（点右上角X）
        if (getWindowProperty("Optimized Matches", WND_PROP_VISIBLE) < 1) {
            break;
        }
    }

    return 0;
}
