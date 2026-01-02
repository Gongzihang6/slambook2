//
// Created by Xiang on 2017/12/19.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "../LK1.png";  // first image
string file_2 = "../LK2.png";  // second image

// 光流跟踪法接口定义
class OpticalFlowTracker {
public:
    // 构造函数
    OpticalFlowTracker(const Mat &img1_,
                        const Mat &img2_,
                        const vector<KeyPoint> &kp1_,
                        vector<KeyPoint> &kp2_,
                        vector<bool> &success_,
                        bool inverse_ = true, bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_) {}

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_guess = false
);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * 因为光流法中，我们计算出的像素运动(dx,dy)通常是浮点数，这意味着下一帧中的关键点可能会
 * 跑到两个像素之间，回了获取非整数坐标处的灰度值，必须使用插值
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // 1. 边界检查：防止索引超出图像范围
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    // 2. 计算小数部分 (xx, yy) 和整数部分 (floor)
    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    // 3. 双线性插值公式
    // 公式：f(x,y) ≈ (1-xx)(1-yy)I(0,0) + xx(1-yy)I(1,0) + (1-xx)yyI(0,1) + xxyyI(1,1)
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main(int argc, char **argv) {

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    /**
     * 500（maxCorners)：如果检测到了 1000 个符合条件的角点，只保留分数最高的前 500 个。这对于 SLAM 很重要，因为特征点太多会拖慢计算速度，太少又容易丢失跟踪。
     * 0.01 (qualityLevel): 假设全图中所有像素算出来的 Shi-Tomasi 分数最高为 $S_{max}$（比如 1500）。
     *                      那么，所有分数低于 $1500 \times 0.01 = 15$ 的点，直接被认为是“非角点”或“弱角点”而被丢弃。
     * 20 (minDistance): 为了保证特征点在图像中分布均匀，不允许两个角点靠得太近。如果你在坐标 $(100, 100)$ 找到了一个强角点，
     *                  那么以它为圆心，半径 20 像素范围内，禁止出现第二个角点（即使那个点分数也很高）。这可以有效避免“特征扎堆”现象，让光流追踪更鲁棒。
     */
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}


 void OpticalFlowSingleLevel(const Mat &img1,
                            const Mat &img2,
                            const vector<KeyPoint> &kp1,
                            vector<KeyPoint> &kp2,
                            vector<bool> &success,
                            bool inverse, bool has_initial) {
    kp2.resize(kp1.size());     // 首先根据图1中关键点的数量，给图2的关键点 kp2 和状态 success 分配空间。这是为了防止多线程写入时发生内存冲突或频繁的动态扩容。
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);    // 包装类，用来把所有需要的数据（图片、点）打包传给线程
    // parallel_for_ 是opencv提供的并行计算接口，它会自动把 0 到 kp1.size() 这个范围切分成若干个小块（Range），分配给 CPU 的不同核心去跑
    // std::bind: 把 calculateOpticalFlow 成员函数和 tracker 对象绑定在一起，作为任务函数传给线程
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

/**
 * 光流法核心，将每个关键点的追踪看作一个非线性最小二乘问题
 */
void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {     // 实现OpticalFlowTracker接口的方法
    // parameters
    int half_patch_size = 4;    // 窗口大小，这里取 4，意味着通过 9x9 (4+1+4) 的窗口来计算光流
    int iterations = 10;        // 高斯-牛顿法的最大迭代次数

    // 遍历分配给当前线程的关键点 (parallel_for_ 分配的 range)
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias
        Eigen::Vector2d J;  // jacobian
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) {
                // 正向法：每次迭代 J 都会变，所以 H 也会变，必须清零重新累加
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // 反向法：H 是固定的（只跟第一张图有关），算过一次后就不变了，所以不需要清零 H
                // 但是 b 包含误差 error，error 每次都会变，所以 b 必须清零
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;   // 记录当前的总误差平方和，用于判断是否发散

            // compute cost and jacobian
            // 遍历窗口内所有像素
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    // 1. 计算光度误差 (Error)
                    // Error = I1(原始位置) - I2(估计位置)
                    // GetPixelValue 使用双线性插值，因为 kp.pt + dx 可能是小数坐标
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian

                    // 【关键分支】根据是否使用反向法来决定是否重置 H
                    if (inverse == false) {
                        // 【正向法】：在图2 (img2) 的当前估计位置 (kp+dx) 处计算梯度
                        // 使用中心差分法：(右-左)/2, (下-上)/2
                        // 这里的 -1.0 来自公式推导：Error = I1 - I2，对 dx 求导会带出一个负号
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } 
                    else if (iter == 0) {   // 只有在inverse为true且第一次迭代时才计算 J
                        // 【反向法】：在图1 (img1) 的原始位置 (kp) 处计算梯度
                        // 这里的关键优化：J 只计算一次 (iter==0时)！
                        // 因为图1的坐标 x,y 是固定的，dx,dy 的变化不影响这里。
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }

            // compute update
            // 4. 求解线性方程 H * update = b
            // 使用 LDLT 分解法求解，比求逆矩阵更快更稳
            Eigen::Vector2d update = H.ldlt().solve(b);

            // 【异常检查 1】如果 H 不可逆（例如纯白/纯黑区域，没有梯度），解出来可能是 NaN
            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;   // 标记追踪失败
                break;
            }
            
            // 【异常检查 2】如果这一步的误差比上一步还大，说明“走过头了”或发散了
            if (iter > 0 && cost > lastCost) {
                break;      // 停止迭代，保留上一次较好的结果
            }

            // 5. 更新估计值
            dx += update[0];
            dy += update[1];
            lastCost = cost;    // 记录当前误差，供下一次比较
            succ = true;        // 暂时标记为成功

            // 【收敛判断】如果更新量极小（小于 0.01 像素），说明已经收敛，不需要再算了
            if (update.norm() < 1e-2) {
                // converge
                break;
            }
        }
        // 循环结束后，保存最终结果，标记当前迭代状态成功
        success[i] = succ;

        // set kp2
        // 将最终计算出的位移 (dx, dy) 叠加到原始坐标上，得到图2中的坐标
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}

/**
 * 这段代码实现了多层金字塔光流法（Multi-Level Optical Flow）。
 * 它的核心思想是**“由粗到精”（Coarse-to-Fine）**：
 *      先在图像的缩小版（金字塔顶层）上计算光流。因为图像缩小了，原本的大运动在缩小图上也变成了小运动，满足了LK光流“小运动”的假设。
 *      将算出的粗略运动作为初始猜测值（Initial Guess），传递给下一层（较大一点的图）。
 *      在下一层上进行精细化调整，直到回到原图。
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) {

    // parameters
    int pyramids = 4;       // 金字塔层数：4层 (0, 1, 2, 3)
    double pyramid_scale = 0.5;     // 缩放比例：每一层是上一层的一半
    double scales[] = {1.0, 0.5, 0.25, 0.125};      // 预计算好的比例数组，对应层级 0,1,2,3

    // create pyramids
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2;     // 用于存放两张图的金字塔图像序列
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            // 第 0 层就是原图
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } 
        else {
            Mat img1_pyr, img2_pyr;
            // 使用 cv::resize 将上一层(i-1)图像的长宽各缩小一半
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    
    // coarse-to-fine LK tracking in pyramids
    // 在开始计算前，需要把关键点的坐标变换到金字塔的最顶层（最小的那张图）。
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp;

        // 将原图坐标乘以最顶层的缩放比例 (0.125)
        // 例如原图 (800, 800) -> 顶层 (100, 100)
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);      // 初始猜测：假设在顶层没动，kp2的初始位置等于kp1
    }

    // 算法的灵魂。从最顶层（Level 3）开始算，一直算到第 0 层。
    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();
        // 【关键调用】调用单层光流接口
        // 参数解释：
        // pyr1[level], pyr2[level]: 当前层级的图像
        // kp1_pyr, kp2_pyr: 当前层级的坐标
        // inverse: 是否使用反向法加速
        // true (最后一个参数): 表示 has_initial_guess = true！这非常重要！
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        // 如果没到最大的图，则需要把坐标放大，传递给下一层
        // 这就好比你找路：先看世界地图找到大方向（粗），再看城市地图找街道（细），最后看街区地图找门牌号（精）
        if (level > 0) {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;     // 坐标放大 2 倍
        }
    }

    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);      // // 循环结束后，kp2_pyr 里存的就是 Level 0 (原图) 的最终坐标
}
