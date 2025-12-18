/**
 * Haar Feature Implementation based on Journey2SLAM
 * * 功能：
 * 1. 手动实现积分图计算 (Integral Image Calculation)
 * 2. 手动实现 Haar 特征模板定义与计算 (Haar Feature Extraction)
 * 3. 演示功能 1：单张图片上的 Haar 特征检测（寻找最大响应区域）并可视化
 * 4. 演示功能 2：两张图片基于 Haar 特征向量的匹配
 * * 编译说明 (需链接 OpenCV):
 * g++ main.cpp -o haar_demo `pkg-config --cflags --libs opencv4`
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

// ==========================================
// 1. 基础数据结构与积分图算法
// ==========================================

// 定义 Haar 特征中的一个矩形区域
struct HaarRect {
    int x, y, w, h; // 矩形相对于特征左上角的偏移和宽高
    float weight;   // 权重 (白色区域为正，黑色区域为负)
};

// 定义一个完整的 Haar 特征 (由多个矩形组成)
struct HaarFeature {
    string name;
    int width, height; // 特征模板的总大小
    vector<HaarRect> rects;
};

// 积分图类：手动实现积分图构建与区域求和
class IntegralImage {
public:
    int width, height;
    vector<vector<double>> data; // 使用 double 防止溢出，存储积分图

    IntegralImage() {}

    // 根据输入图像构建积分图
    // 参考公式：
    // s(x,y) = s(x,y-1) + I(x,y)   (行元素累加，这里原文公式里的x,y可能指行列，我们代码中x为列，y为行)
    // I_i(x,y) = I_i(x-1,y) + s(x,y)
    void compute(const Mat& img) {
        CV_Assert(img.channels() == 1); // 必须是灰度图
        height = img.rows;
        width = img.cols;
        
        // 初始化积分图矩阵，大小与原图一致 (或者+1 padding，这里为了严格对应公式使用原大小+边界判断)
        data.resize(height, vector<double>(width, 0.0));

        // 临时变量 s(x,y)，这里我们需要一个数组来存储当前行的列累加值，或者直接在循环中维护
        // 按照博客中的迭代公式：
        // s(x, y) 表示第 x 列，从 0 到 y 行的像素和 (原文里 x, y 坐标定义可能与 OpenCV 相反，这里按常规 x=col, y=row)
        // Let's implement exactly as typical integral image logic but manually.
        
        for (int y = 0; y < height; y++) {
            double row_sum = 0; // 相当于公式中的 s(x,y) 在遍历 x 时的当前值
            for (int x = 0; x < width; x++) {
                double val = (double)img.at<uchar>(y, x);
                
                row_sum += val; // s(x,y) = s(x-1, y) + i(x,y) -> 这里稍微调整了方向理解，row_sum 是当前行的累加
                
                // I_i(x,y) = I_i(x, y-1) + s(x,y) (当前位置积分 = 上一行同位置积分 + 当前行累加)
                double above_sum = (y > 0) ? data[y - 1][x] : 0.0;
                
                data[y][x] = above_sum + row_sum;
            }
        }
    }

    // 计算任意矩形区域的像素和
    // 公式：Sum(D) = I(x4,y4) - I(x2,y2) - I(x3,y3) + I(x1,y1)
    // 对应坐标：(x,y) 是左上角，(x+w-1, y+h-1) 是右下角
    double getSum(int x, int y, int w, int h) const {
        if (x < 0 || y < 0 || x + w > width || y + h > height || w <= 0 || h <= 0) return 0.0;

        int x_right = x + w - 1;
        int y_bottom = y + h - 1;
        int x_left = x - 1;
        int y_top = y - 1;

        double A = (x_left >= 0 && y_top >= 0) ? data[y_top][x_left] : 0.0;
        double B = (y_top >= 0) ? data[y_top][x_right] : 0.0;
        double C = (x_left >= 0) ? data[y_bottom][x_left] : 0.0;
        double D = data[y_bottom][x_right];

        return D - B - C + A;
    }
};

// ==========================================
// 2. Haar 特征计算核心
// ==========================================

// 计算一个 Haar 特征在图像 (off_x, off_y) 位置的响应值
// scale 用于缩放特征模板
double computeHaarResponse(const IntegralImage& ii, const HaarFeature& feature, int off_x, int off_y, float scale = 1.0) {
    double sum = 0.0;
    
    // 遍历特征包含的所有矩形 (例如：一个黑色矩形，一个白色矩形)
    for (const auto& r : feature.rects) {
        // 根据 scale 缩放矩形
        int w_s = (int)(r.w * scale);
        int h_s = (int)(r.h * scale);
        int x_s = off_x + (int)(r.x * scale);
        int y_s = off_y + (int)(r.y * scale);

        // 利用积分图快速计算区域和
        double rect_sum = ii.getSum(x_s, y_s, w_s, h_s);
        
        // 累加：区域和 * 权重
        sum += rect_sum * r.weight;
    }
    return sum;
}

// 创建标准的 Haar 特征模板
// 参考博客图片：两矩形特征（水平/垂直）、三矩形特征
vector<HaarFeature> createStandardHaarFeatures() {
    vector<HaarFeature> features;

    // 1. 水平边缘特征 (左白右黑)
    //  [W][B]
    HaarFeature h_edge;
    h_edge.name = "Horizontal Edge";
    h_edge.width = 2; h_edge.height = 1;
    h_edge.rects.push_back({0, 0, 1, 1, 1.0f});  // White
    h_edge.rects.push_back({1, 0, 1, 1, -1.0f}); // Black
    features.push_back(h_edge);

    // 2. 垂直边缘特征 (上白下黑)
    //  [W]
    //  [B]
    HaarFeature v_edge;
    v_edge.name = "Vertical Edge";
    v_edge.width = 1; v_edge.height = 2;
    v_edge.rects.push_back({0, 0, 1, 1, 1.0f});  // White
    v_edge.rects.push_back({0, 1, 1, 1, -1.0f}); // Black
    features.push_back(v_edge);

    // 3. 垂直线条特征 (白-黑-白)
    //  [W]
    //  [B]
    //  [W]
    HaarFeature v_line;
    v_line.name = "Vertical Line";
    v_line.width = 1; v_line.height = 3;
    v_line.rects.push_back({0, 0, 1, 1, 1.0f});  // White
    v_line.rects.push_back({0, 1, 1, 1, -2.0f}); // Black (权重-2以平衡面积)
    v_line.rects.push_back({0, 2, 1, 1, 1.0f});  // White
    features.push_back(v_line);
    
    // 4. 对角特征 (棋盘)
    // [W][B]
    // [B][W]
    HaarFeature diag;
    diag.name = "Diagonal";
    diag.width = 2; diag.height = 2;
    diag.rects.push_back({0, 0, 1, 1, 1.0f}); 
    diag.rects.push_back({1, 0, 1, 1, -1.0f});
    diag.rects.push_back({0, 1, 1, 1, -1.0f});
    diag.rects.push_back({1, 1, 1, 1, 1.0f});
    features.push_back(diag);

    return features;
}

// ==========================================
// 3. 辅助绘制函数
// ==========================================

// 在图像上绘制 Haar 特征
// 白色区域画白框填充，黑色区域画黑框填充
void drawHaarFeature(Mat& img, const HaarFeature& feature, int x, int y, float scale) {
    for (const auto& r : feature.rects) {
        int w_s = (int)(r.w * scale);
        int h_s = (int)(r.h * scale);
        int x_s = x + (int)(r.x * scale);
        int y_s = y + (int)(r.y * scale);
        
        Scalar color = (r.weight > 0) ? Scalar(255) : Scalar(0); // 白或黑
        rectangle(img, Rect(x_s, y_s, w_s, h_s), color, 2); 
        // 为了显示清楚，填充半透明颜色
        Mat roi = img(Rect(x_s, y_s, w_s, h_s));
        double alpha = 0.5;
        addWeighted(roi, alpha, Mat(roi.size(), roi.type(), color), 1.0 - alpha, 0.0, roi);
    }
}

// ==========================================
// 4. 任务实现：检测与匹配
// ==========================================

// 任务1：输入图片，检测某个 Haar 特征（寻找最大响应）并可视化
void detectAndVisualize(const Mat& src, const HaarFeature& feature, float scale) {
    Mat img_gray;
    if (src.channels() == 3) cvtColor(src, img_gray, COLOR_BGR2GRAY);
    else img_gray = src.clone();

    // 1. 构建积分图
    IntegralImage ii;
    ii.compute(img_gray);

    // 2. 遍历图像，计算特征值
    double max_response = -1e9;
    Point max_loc(0, 0);

    // 特征实际大小
    int feat_w = (int)(feature.width * scale);
    int feat_h = (int)(feature.height * scale);

    for (int y = 0; y < img_gray.rows - feat_h; y += 2) { // 步长2加快速度
        for (int x = 0; x < img_gray.cols - feat_w; x += 2) {
            double val = abs(computeHaarResponse(ii, feature, x, y, scale)); // 取绝对值寻找强响应
            if (val > max_response) {
                max_response = val;
                max_loc = Point(x, y);
            }
        }
    }

    // 3. 可视化
    Mat result = src.clone();
    if (result.channels() == 1) cvtColor(result, result, COLOR_GRAY2BGR);

    cout << "Detected " << feature.name << " at " << max_loc << " with response " << max_response << endl;
    
    // 画出特征所在的矩形位置
    drawHaarFeature(result, feature, max_loc.x, max_loc.y, scale);
    
    // 画一个醒目的外框
    rectangle(result, Rect(max_loc.x, max_loc.y, feat_w, feat_h), Scalar(0, 0, 255), 1);
    putText(result, "Max Haar Response", Point(max_loc.x, max_loc.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

    imshow("Haar Detection Visualization", result);
}

// 任务2：可视化两张图片的 Haar 特征匹配
// 由于 Haar 通常不是点特征，我们手动构建一个 "Haar 描述子"：
// 在网格点上，计算一组不同 Haar 特征的响应值，作为该点的特征向量，然后进行匹配。
void matchHaarFeatures(const Mat& img1, const Mat& img2) {
    Mat gray1, gray2;
    if (img1.channels() == 3) cvtColor(img1, gray1, COLOR_BGR2GRAY); else gray1 = img1.clone();
    if (img2.channels() == 3) cvtColor(img2, gray2, COLOR_BGR2GRAY); else gray2 = img2.clone();

    IntegralImage ii1, ii2;
    ii1.compute(gray1);
    ii2.compute(gray2);

    vector<HaarFeature> templates = createStandardHaarFeatures();
    
    // 定义网格点
    int step = 40; 
    float scale = 12.0; // 特征尺度
    
    struct KeyPointDesc {
        Point pt;
        vector<double> descriptor;
    };

    auto computeDescriptors = [&](const IntegralImage& ii, int rows, int cols) -> vector<KeyPointDesc> {
        vector<KeyPointDesc> kps;
        for (int y = step; y < rows - step; y += step) {
            for (int x = step; x < cols - step; x += step) {
                KeyPointDesc kp;
                kp.pt = Point(x, y);
                // 计算该点周围的一组 Haar 特征响应作为描述子
                // 我们在点周围偏移特征以居中
                for (const auto& feat : templates) {
                    int fw = (int)(feat.width * scale);
                    int fh = (int)(feat.height * scale);
                    // 简单的归一化：除以面积
                    double resp = computeHaarResponse(ii, feat, x - fw/2, y - fh/2, scale) / (fw*fh); 
                    kp.descriptor.push_back(resp);
                }
                kps.push_back(kp);
            }
        }
        return kps;
    };

    vector<KeyPointDesc> desc1 = computeDescriptors(ii1, gray1.rows, gray1.cols);
    vector<KeyPointDesc> desc2 = computeDescriptors(ii2, gray2.rows, gray2.cols);

    // 暴力匹配 (Brute Force Matching)
    vector<DMatch> matches;
    for (size_t i = 0; i < desc1.size(); i++) {
        double min_dist = 1e9;
        int best_idx = -1;
        
        for (size_t j = 0; j < desc2.size(); j++) {
            // 计算欧氏距离
            double dist = 0.0;
            for (size_t k = 0; k < desc1[i].descriptor.size(); k++) {
                double diff = desc1[i].descriptor[k] - desc2[j].descriptor[k];
                dist += diff * diff;
            }
            dist = sqrt(dist);

            if (dist < min_dist) {
                min_dist = dist;
                best_idx = j;
            }
        }
        
        // 简单的阈值过滤
        if (min_dist < 50.0) { // 阈值取决于像素值范围，需根据实际情况调整
             matches.push_back(DMatch((int)i, best_idx, (float)min_dist));
        }
    }

    // 绘制匹配结果
    Mat match_img;
    Mat kps1_cv, kps2_cv; // 需要转换为 KeyPoint 格式才能用 drawMatches
    vector<KeyPoint> kp1_vec, kp2_vec;
    for(auto& d : desc1) kp1_vec.push_back(KeyPoint(d.pt, step));
    for(auto& d : desc2) kp2_vec.push_back(KeyPoint(d.pt, step));

    drawMatches(img1, kp1_vec, img2, kp2_vec, matches, match_img, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // 为了更直观，我们在图上画出被匹配的一个特征的样子（取第一个匹配点）
    if (!matches.empty()) {
        Point pt1 = desc1[matches[0].queryIdx].pt;
        // 在匹配图的左侧画出一个 Vertical Line 特征示意
        drawHaarFeature(match_img, templates[2], pt1.x - 6, pt1.y - 18, scale);
        putText(match_img, "Haar Features Used for Matching", Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    }

    imshow("Haar Feature Matching (Manual Descriptor)", match_img);
}

int main(int argc, char** argv) {
    // 读取图像 (请替换为你本地的图片路径，或者传入参数)
    // 为了演示方便，这里如果读不到图片会生成简单的测试图
    if (argc < 3) {
        cerr << "Usage: ./Haar_self <img1> <img2>" << endl;
        return -1;
    }

    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);   // 以灰度图方式加载图片
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cout << "未找到图片，生成测试图片..." << endl;
        img1 = Mat::zeros(400, 400, CV_8UC1);
        img2 = Mat::zeros(400, 400, CV_8UC1);
        // 画一些形状
        rectangle(img1, Rect(100, 100, 100, 100), Scalar(200), -1); // 亮块
        rectangle(img2, Rect(120, 110, 100, 100), Scalar(200), -1); // 稍微移动的亮块
        line(img1, Point(50, 50), Point(50, 200), Scalar(255), 5); // 线条
        line(img2, Point(70, 50), Point(70, 200), Scalar(255), 5);
    }

    // 1. 演示单图 Haar 特征检测
    // 使用 "Vertical Line" 特征，寻找图中的垂直亮线条
    vector<HaarFeature> feats = createStandardHaarFeatures();
    cout << "正在进行 Haar 特征检测..." << endl;
    detectAndVisualize(img1, feats[2], 20.0); // scale=20, 比较大的特征

    // 2. 演示两图 Haar 特征匹配
    cout << "正在进行 Haar 特征匹配..." << endl;
    matchHaarFeatures(img1, img2);

    cout << "按任意键退出..." << endl;
    waitKey(0);

    return 0;
}
