/**
 * 功能：手动实现 Harris 角点检测 (Manual Harris Corner Detection)
 * 作者：Gemini (For Math Master User)
 * 日期：2025-12-19
 * * 实现原理步骤：
 * 1. 灰度化 (Grayscale)
 * 2. 计算图像梯度 I_x, I_y (Derivatives using Sobel)
 * 3. 计算结构张量 M 的原始分量: I_x^2, I_y^2, I_x*I_y
 * 4. 高斯加权 (Window Function): 对分量进行高斯模糊，得到 M = [A C; C B]
 * 5. 计算角点响应值 R = det(M) - k * (trace(M))^2
 * 6. 阈值截断 & 非极大值抑制 (Thresholding & Non-Maximum Suppression)
 * 7. 可视化
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Harris 参数定义
const double HARRIS_K = 0.04;      // 经验常数 alpha，通常取 0.04 - 0.06
const int WINDOW_SIZE = 3;         // 高斯窗口大小 (3x3 或 5x5)
const double SIGMA = 1.0;          // 高斯函数的标准差

void manualHarrisCorner(const Mat& src, Mat& outputViz) {
    // 1. 预处理：转为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 转为浮点型，防止计算梯度和乘积时溢出
    Mat grayFloat;
    gray.convertTo(grayFloat, CV_64F);

    // 2. 计算梯度 I_x 和 I_y
    // 这里完全手动构建 Sobel 核，不隐藏细节
    Mat Ix, Iy;
    // Sobel X 核
    Mat sobelX = (Mat_<double>(3, 3) << -1, 0, 1, 
                                        -2, 0, 2, 
                                        -1, 0, 1);
    // Sobel Y 核
    Mat sobelY = (Mat_<double>(3, 3) << -1, -2, -1, 
                                         0,  0,  0, 
                                         1,  2,  1);

    // 使用 filter2D 进行卷积操作计算梯度
    filter2D(grayFloat, Ix, CV_64F, sobelX);
    filter2D(grayFloat, Iy, CV_64F, sobelY);

    // 3. 计算结构张量 M 的三个分量 (未加权前)
    // M = sum [ Ix^2   IxIy ]
    //         [ IxIy   Iy^2 ]
    Mat Ix2, Iy2, Ixy;
    multiply(Ix, Ix, Ix2);      // Ix^2
    multiply(Iy, Iy, Iy2);      // Iy^2
    multiply(Ix, Iy, Ixy);      // Ix * Iy

    // 4. 高斯加权 (Window Function w(u,v))
    // 这一步对应公式中的求和符号 sum_w，使用高斯核赋予中心像素更高权重
    Mat A, B, C;
    GaussianBlur(Ix2, A, Size(WINDOW_SIZE, WINDOW_SIZE), SIGMA); // A = sum(w * Ix^2)
    GaussianBlur(Iy2, B, Size(WINDOW_SIZE, WINDOW_SIZE), SIGMA); // B = sum(w * Iy^2)
    GaussianBlur(Ixy, C, Size(WINDOW_SIZE, WINDOW_SIZE), SIGMA); // C = sum(w * IxIy)

    // 5. 计算 Harris 响应值 R
    // R = det(M) - k * trace(M)^2
    // det(M) = A*B - C*C
    // trace(M) = A + B
    
    Mat R_map = Mat::zeros(src.size(), CV_64F);
    double max_R = -1e30; // 记录最大响应值，用于设置相对阈值

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            double a_val = A.at<double>(r, c);
            double b_val = B.at<double>(r, c);
            double c_val = C.at<double>(r, c);

            double det_M = a_val * b_val - c_val * c_val;
            double trace_M = a_val + b_val;

            double R = det_M - HARRIS_K * pow(trace_M, 2);
            
            R_map.at<double>(r, c) = R;

            if (R > max_R) {
                max_R = R;
            }
        }
    }

    // 6. 阈值处理与非极大值抑制 (NMS)
    // 只有 R > threshold 且 R 是局部最大值时，才认为是角点
    double threshold = 0.01 * max_R; // 使用相对阈值，鲁棒性更好
    outputViz = src.clone();

    // 为了避免边界越界，从 1 遍历到 rows-1
    for (int r = 1; r < src.rows - 1; r++) {
        for (int c = 1; c < src.cols - 1; c++) {
            double current_R = R_map.at<double>(r, c);

            // 阈值判断：必须是正数且足够大
            if (current_R > threshold) {
                // 非极大值抑制 (NMS)：检查 3x3 邻域
                bool isLocalMax = true;
                
                // 遍历周围 8 个像素
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        if (dr == 0 && dc == 0) continue; // 跳过中心点
                        if (R_map.at<double>(r + dr, c + dc) > current_R) {
                            isLocalMax = false;
                            break;
                        }
                    }
                    if (!isLocalMax) break;
                }

                // 如果是局部极大值，画出来
                if (isLocalMax) {
                    // 绘制红色圆圈标记角点
                    circle(outputViz, Point(c, r), 4, Scalar(0, 0, 255), 2, 8, 0);
                }
            }
        }
    }
}

int main() {
    // 读取图像 (请修改为你本地的图片路径)
    string imgPath = "../g1.png";   // ../g1.png
    // 如果没有图片，创建一个简单的测试图：黑色背景，白色矩形
    Mat src = Mat::zeros(400, 400, CV_8UC3);
    rectangle(src, Point(100, 100), Point(300, 300), Scalar(255, 255, 255), -1);
    // 旋转一下矩形，制造更多不规则角点 (可选)
    Point2f center(200, 200);
    Mat rot = getRotationMatrix2D(center, 45, 1.0);
    warpAffine(src, src, rot, Size(400, 400));

    // 如果想读取真实图片，取消下面注释
    src = imread(imgPath); 
    if (src.empty()) { cout << "无法读取图像" << endl; return -1; }

    Mat result;
    cout << "开始进行 Harris 角点检测..." << endl;
    
    manualHarrisCorner(src, result);

    // 显示结果
    imshow("Original Image", src);
    imshow("Harris Manual Implementation", result);

    cout << "检测完成。按任意键退出。" << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}