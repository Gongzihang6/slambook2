/**
 * 手写 SURF (Speeded-Up Robust Features) 核心原理演示
 * 重点实现：积分图、Fast-Hessian 矩阵近似、尺度空间构建、非极大值抑制
 * 作者: Gemini for User (Math Master Student)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

// 定义一个简单的结构体来存储特征点
struct SurfPoint {
    float x, y;     // 坐标
    float scale;    // 尺度 (sigma)
    float response; // Hessian 行列式响应值
    int octave;     // 所在的组
    int layer;      // 所在的层
};

// ==========================================
// 1. 积分图辅助函数 (Integral Image)
// 原理：I_sum(x,y) = sum(I(i,j)) for i<=x, j<=y
// 利用积分图，任意矩形区域的像素和可以在 O(1) 时间内计算：
// Area = A + D - B - C
// ==========================================
float boxIntegral(const Mat& integral_img, int r, int c, int rows, int cols) {
    // r, c: 矩形左上角坐标
    // rows, cols: 矩形的高度和宽度
    
    // 边界检查
    int r1 = min(max(r, 0), integral_img.rows - 1);
    int c1 = min(max(c, 0), integral_img.cols - 1);
    int r2 = min(max(r + rows, 0), integral_img.rows - 1);
    int c2 = min(max(c + cols, 0), integral_img.cols - 1);

    // A: (r1, c1), B: (r1, c2), C: (r2, c1), D: (r2, c2)
    // 注意：OpenCV的integral计算结果比原图多一行一列，(0,0)是0。
    // 因此查找 (r,c) 实际上是在积分图中找 (r,c)。
    // 这里的 boxIntegral 假设输入已经是 compute 好的积分图。
    
    float A = integral_img.at<float>(r1, c1);
    float B = integral_img.at<float>(r1, c2);
    float C = integral_img.at<float>(r2, c1);
    float D = integral_img.at<float>(r2, c2);

    return max(0.0f, A + D - B - C);
}

// ==========================================
// 2. Fast Hessian 响应计算
// 原理：计算 Hessian 矩阵的行列式 det(H) = Dxx * Dyy - (0.9 * Dxy)^2
// 使用箱式滤波器近似二阶高斯微分。
// ==========================================
void buildResponseLayer(const Mat& integral_img, Mat& response_map, int filter_size, int step) {
    int rows = integral_img.rows - 1; // 原图高度
    int cols = integral_img.cols - 1; // 原图宽度
    
    response_map = Mat::zeros(rows, cols, CV_32F);
    
    // 滤波器参数 (基于 SURF 论文中 9x9 模板的比例)
    // L_xx 模板包含三个矩形：左(-1)、中(2)、右(-1)
    //  lobe (波瓣) 的大小约为 filter_size / 3
    int w = filter_size / 3; 
    
    // 归一化系数，为了消除滤波器大小变化带来的影响
    // 论文公式: 1 / (L^2) 
    float inverse_area_norm = 1.0f / (filter_size * filter_size); 

    // 遍历图像 (以 step 为步长加速)
    // 这里的 margin 是为了防止滤波器越界
    int margin = filter_size / 2;
    
    for (int r = margin; r < rows - margin; r += step) {
        for (int c = margin; c < cols - margin; c += step) {
            
            // --- 计算 Dxx ---
            // 9x9 模板中：高度9，宽度3的黑白黑条纹
            // 左边(负): r-4, c-4, 9x3
            // 中间(正): r-4, c-1, 9x3
            // 右边(负): r-4, c+2, 9x3
            // 下面的公式是通用的比例推导
            float Dxx = boxIntegral(integral_img, r - w*1.5, c - w/2, 3*w-1, w) * (-3.0f); // 这是一个简化写法，实际需要分别计算三个块
            
            // 为了完全还原原理，我们严格按照 SURF 的 9x9 布局比例来写 Dxx
            // Box 1 (左, 负权重): y:[-L/2, L/2], x:[-L/2, -L/6]
            float v1 = boxIntegral(integral_img, r - (filter_size/2), c - (filter_size/2), filter_size, w);
            // Box 2 (中, 正权重): y:[-L/2, L/2], x:[-L/6, L/6] (权重通常是2倍或3倍以平衡，这里用简化版：中间减两边)
            float v2 = boxIntegral(integral_img, r - (filter_size/2), c - (filter_size/2) + w, filter_size, w);
            // Box 3 (右, 负权重)
            float v3 = boxIntegral(integral_img, r - (filter_size/2), c - (filter_size/2) + 2*w, filter_size, w);
            // SURF 论文中 Dxx 权重：中间部分 * 3 - 整体? 
            // 实际上常用近似： Dxx = (v2 * 3) - (v1 + v2 + v3); (保持总和为0)
            Dxx = (v2 * 3.0f) - (v1 + v2 + v3);

            // --- 计算 Dyy (逻辑同 Dxx，只是旋转90度) ---
            float h1 = boxIntegral(integral_img, r - (filter_size/2), c - (filter_size/2), w, filter_size);
            float h2 = boxIntegral(integral_img, r - (filter_size/2) + w, c - (filter_size/2), w, filter_size);
            float h3 = boxIntegral(integral_img, r - (filter_size/2) + 2*w, c - (filter_size/2), w, filter_size);
            float Dyy = (h2 * 3.0f) - (h1 + h2 + h3);

            // --- 计算 Dxy ---
            // 四个角落的方块
            // 左上(1), 右上(-1), 左下(-1), 右下(1)
            // 块大小约为 w x w
            int d = 1; // 对角距离微调，通常是 w
            float tl = boxIntegral(integral_img, r - w - d, c - w - d, w, w); // Top-Left
            float tr = boxIntegral(integral_img, r - w - d, c + d, w, w);     // Top-Right
            float bl = boxIntegral(integral_img, r + d, c - w - d, w, w);     // Bottom-Left
            float br = boxIntegral(integral_img, r + d, c + d, w, w);         // Bottom-Right
            float Dxy = (tl + br) - (tr + bl);

            // --- 归一化 ---
            Dxx *= inverse_area_norm;
            Dyy *= inverse_area_norm;
            Dxy *= inverse_area_norm;

            // --- Hessian 行列式 ---
            // 0.9 (精确说是 0.912) 是用于平衡 Box Filter 近似带来的误差
            float detH = (Dxx * Dyy) - (0.81f * Dxy * Dxy);

            // 存入响应图
            if (detH > 0) {
                response_map.at<float>(r, c) = detH;
            }
        }
    }
}

// ==========================================
// 3. 非极大值抑制 (NMS) 3x3x3
// 在当前层、上一层、下一层 的 26 邻域中寻找极值
// ==========================================
bool isMaximum(const vector<Mat>& layers, int o, int l, int r, int c) {
    float val = layers[l].at<float>(r, c);
    if (val <= 0) return false;

    // 检查当前层 8 邻域
    for (int rr = -1; rr <= 1; ++rr) {
        for (int cc = -1; cc <= 1; ++cc) {
            if (rr == 0 && cc == 0) continue;
            if (layers[l].at<float>(r + rr, c + cc) >= val) return false;
        }
    }

    // 检查上一层 (如果存在)
    if (l > 0) {
        for (int rr = -1; rr <= 1; ++rr) {
            for (int cc = -1; cc <= 1; ++cc) {
                if (layers[l - 1].at<float>(r + rr, c + cc) >= val) return false;
            }
        }
    }

    // 检查下一层 (如果存在)
    if (l < layers.size() - 1) {
        for (int rr = -1; rr <= 1; ++rr) {
            for (int cc = -1; cc <= 1; ++cc) {
                if (layers[l + 1].at<float>(r + rr, c + cc) >= val) return false;
            }
        }
    }
    return true;
}

// ==========================================
// 主类：MySurfDetector
// ==========================================
class MySurfDetector {
public:
    // 配置参数
    int octaves = 3;     // 金字塔组数
    int layers = 4;      // 每组层数
    float threshold = 500.0f; // Hessian 阈值
    
    void detect(const Mat& img, vector<SurfPoint>& keypoints) {
        // 1. 计算积分图
        Mat integral_img;
        // OpenCV 的 integral 函数会将结果存为 CV_32S 或 CV_64F，这里用 32F 方便计算
        // 积分图的大小是 (rows+1) x (cols+1)
        integral(img, integral_img, CV_32F);    // 32位浮点数，避免大图求和时的数值溢出

        // 2. 构建尺度空间 (Scale Space)
        // SURF 不对图像降采样，而是增大滤波器尺寸
        // Octave 0: 9, 15, 21, 27
        // Octave 1: 15, 27, 39, 51 (步长翻倍)
        // ...
        
        vector<vector<Mat>> response_pyramid;   // 响应金字塔
        vector<vector<int>> filter_sizes;       // 滤波器尺寸序列
        
        cout << "构建 Hessian 响应金字塔..." << endl;
        
        for (int o = 0; o < octaves; ++o) {
            vector<Mat> octave_layers;
            vector<int> octave_sizes;
            int step = 1 << o;      // 当前组octave滤波器尺寸增加步长
            
            // 滤波器尺寸序列定义 (SURF 论文标准)
            // 第一组基础尺寸: 9
            // 每层增加: 6 * (2^o)
            int current_size = 9 + (6 * 2 * o); // 起始并不严格，简化逻辑：
            // 正确逻辑：Size = 9 + 6*k * 2^o
            
            for (int l = 0; l < layers; ++l) {
                int filter_size = 3 + 6 * step * (l + 1); // 9, 15, 21, 27...
                
                Mat response;
                buildResponseLayer(integral_img, response, filter_size, step);
                octave_layers.push_back(response);
                octave_sizes.push_back(filter_size);
                
                // cout << "Octave " << o << " Layer " << l << ": Filter Size " << filter_size << endl;
            }
            response_pyramid.push_back(octave_layers);
            filter_sizes.push_back(octave_sizes);
        }

        // 3. 非极大值抑制与特征点定位
        cout << "执行非极大值抑制(NMS)..." << endl;
        
        for (int o = 0; o < octaves; ++o) {
            int step = 1 << o;
            // 只能在中间层寻找极值 (第0层和最后一层无法比较上下)
            for (int l = 1; l < layers - 1; ++l) {
                Mat& resp = response_pyramid[o][l];
                
                for (int r = 1; r < resp.rows - 1; ++r) {
                    for (int c = 1; c < resp.cols - 1; ++c) {
                        float val = resp.at<float>(r, c);
                        
                        if (val > threshold) {
                            // 检查 3x3x3 邻域最大值
                            if (isMaximum(response_pyramid[o], o, l, r, c)) {
                                SurfPoint kp;
                                // 坐标需要还原到原图尺度（注意 step）
                                kp.x = c; 
                                kp.y = r; 
                                // 尺度 sigma = 1.2 * (filter_size / 9)
                                kp.scale = 1.2f * filter_sizes[o][l] / 9.0f;
                                kp.response = val;
                                kp.octave = o;
                                kp.layer = l;
                                keypoints.push_back(kp);
                            }
                        }
                    }
                }
            }
        }
    }
};

// ==========================================
// 简化的描述子生成 (Upright SURF)
// 原理：在特征点周围取 20s x 20s 的区域，划分为 4x4 个子区域
// 每个子区域计算 Haar 小波响应 dx, dy
// 特征向量: [sum(dx), sum(dy), sum(|dx|), sum(|dy|)] x 16 = 64维
// ==========================================
void computeDescriptor(const Mat& img, const Mat& integral_img, SurfPoint& kp, vector<float>& descriptor) {
    int scale = (int)kp.scale;
    int cx = (int)kp.x;
    int cy = (int)kp.y;
    
    descriptor.resize(64, 0.0f);
    int idx = 0;

    // 区域半径 20s -> 边长 20s，半边长 10s
    // 为了简化，我们不做旋转 (Upright SURF)，假设主方向向上
    
    // 遍历 4x4 个子区域
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            // 子区域起始坐标 (相对于中心点偏移)
            // 整个大区域是从 -10s 到 +10s
            // 每个子区域大小 5s x 5s
            int r_start = cy - 10 * scale + i * 5 * scale;
            int c_start = cx - 10 * scale + j * 5 * scale;
            
            float sum_dx = 0, sum_dy = 0, sum_abs_dx = 0, sum_abs_dy = 0;

            // 在子区域内采样，步长可以是 scale
            // 这里为了简单，采样 5x5 个点
            for (int rr = 0; rr < 5; ++rr) {
                for (int cc = 0; cc < 5; ++cc) {
                    int sample_r = r_start + rr * scale;
                    int sample_c = c_start + cc * scale;
                    
                    // 计算 Haar 小波 (边长 2s)
                    // dx: 左黑右白
                    int haar_size = 2 * scale;
                    float dx = boxIntegral(integral_img, sample_r, sample_c, haar_size, haar_size/2) * (-1) +
                               boxIntegral(integral_img, sample_r, sample_c + haar_size/2, haar_size, haar_size/2) * (1);
                    
                    // dy: 上黑下白
                    float dy = boxIntegral(integral_img, sample_r, sample_c, haar_size/2, haar_size) * (-1) +
                               boxIntegral(integral_img, sample_r + haar_size/2, sample_c, haar_size/2, haar_size) * (1);
                    
                    // 高斯加权 (简化省略，直接累加)
                    sum_dx += dx;
                    sum_dy += dy;
                    sum_abs_dx += abs(dx);
                    sum_abs_dy += abs(dy);
                }
            }
            
            descriptor[idx++] = sum_dx;
            descriptor[idx++] = sum_dy;
            descriptor[idx++] = sum_abs_dx;
            descriptor[idx++] = sum_abs_dy;
        }
    }
    
    // 归一化描述子 (L2 Norm)
    float norm = 0;
    for (float v : descriptor) norm += v * v;
    norm = sqrt(norm);
    if (norm > 1e-6) {
        for (float& v : descriptor) v /= norm;
    }
}

// ==========================================
// 暴力匹配器 (L2 距离)
// ==========================================
void matchFeatures(const vector<vector<float>>& desc1, const vector<vector<float>>& desc2, 
                   vector<DMatch>& matches) {
    for (size_t i = 0; i < desc1.size(); ++i) {
        float min_dist = FLT_MAX;
        int best_idx = -1;
        
        for (size_t j = 0; j < desc2.size(); ++j) {
            float dist = 0;
            // 计算欧氏距离
            for (int k = 0; k < 64; ++k) {
                float diff = desc1[i][k] - desc2[j][k];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = j;
            }
        }
        
        // 简单的距离阈值筛选
        if (min_dist < 0.3f && best_idx != -1) {
            matches.push_back(DMatch(i, best_idx, min_dist));
        }
    }
}

/**
 * ./SURF_self ../g1.png ../g2.png
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: ./my_surf <img1> <img2>" << endl;
        return -1;
    }

    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);   // 以灰度图方式加载图片
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);
    
    if (img1.empty() || img2.empty()) return -1;    // 如果图像为空，直接退出程序

    // 1. 特征检测
    MySurfDetector detector;
    vector<SurfPoint> kps1_surf, kps2_surf;
    
    double t1 = (double)getTickCount();    // OpenCV 中用于精准计时的一个核心函数，返回从操作系统启动（或者某个特定时间点）到当前时刻所经过的时钟周期数（Tick数）。
    detector.detect(img1, kps1_surf);
    detector.detect(img2, kps2_surf);
    double t2 = (double)getTickCount();
    
    cout << "图1 特征点数: " << kps1_surf.size() << endl;
    cout << "图2 特征点数: " << kps2_surf.size() << endl;
    cout << "检测耗时: " << (t2 - t1) / getTickFrequency() * 1000 << " ms" << endl;

    // 2. 转换数据结构以便后续计算描述子
    // 需要重新计算积分图，因为Detector内部没传出来 (为了代码解耦)
    Mat int1, int2;
    integral(img1, int1, CV_32F);
    integral(img2, int2, CV_32F);
    
    vector<vector<float>> desc1, desc2;
    
    // 计算描述子
    for(auto& kp : kps1_surf) {
        vector<float> d;
        computeDescriptor(img1, int1, kp, d);
        desc1.push_back(d);
    }
    for(auto& kp : kps2_surf) {
        vector<float> d;
        computeDescriptor(img2, int2, kp, d);
        desc2.push_back(d);
    }

    // 3. 匹配
    vector<DMatch> matches;
    matchFeatures(desc1, desc2, matches);
    cout << "匹配点对数量: " << matches.size() << endl;

    // 4. 可视化
    // 将自定义的 SurfPoint 转为 OpenCV 的 KeyPoint 以便使用 drawMatches
    vector<KeyPoint> opencv_kps1, opencv_kps2;
    for(auto& p : kps1_surf) opencv_kps1.push_back(KeyPoint(p.x, p.y, p.scale));
    for(auto& p : kps2_surf) opencv_kps2.push_back(KeyPoint(p.x, p.y, p.scale));
    
    Mat img_matches;
    drawMatches(img1, opencv_kps1, img2, opencv_kps2, matches, img_matches);
    
    // ================== 修改开始 ==================
    string winName = "My Handwritten SURF Matches";
    
    // 1. 创建窗口，并设置为 WINDOW_NORMAL (允许手动或代码调整大小)
    namedWindow(winName, WINDOW_NORMAL);
    
    // 2. 强制设置窗口大小 (例如设置成 1280x720，或者你屏幕合适的大小)
    // 如果想要更智能，可以根据原图比例缩放，例如：
    // resizeWindow(winName, img_matches.cols / 2, img_matches.rows / 2);
    resizeWindow(winName, 1024, 768); 
    
    // 3. 显示图像
    imshow(winName, img_matches);
    // ================== 修改结束 ==================

    waitKey(0);

    return 0;
}
