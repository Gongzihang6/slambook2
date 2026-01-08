#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>  // 计时工具 (已过时，但书中代码用了)

// for sophus: 李代数库，用于处理 SE(3) 位姿
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen: 线性代数库，处理矩阵和向量
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

// for opencv: 图像处理
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

/**********************************************
* 本程序演示了单目相机在已知轨迹下的稠密深度估计
* 使用极线搜索 + NCC 匹配的方式，与书本的 12.2 节对应
* 请注意本程序并不完美，你完全可以改进它——我其实在故意暴露一些问题(这是借口)。
***********************************************/

// ------------------------------------------------------------------
// parameters
const int boarder = 20;         // 边缘宽度，图像边缘的像素通常质量差或无法匹配，直接忽略
const int width = 640;          // 图像宽度
const int height = 480;         // 图像高度
const double fx = 481.2f;       // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;    // NCC匹配窗口半径。实际窗口大小是 (2*3+1) = 7x7
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // 收敛判定：当深度方差小于 0.1 时，认为该像素深度已确信，不再更新
const double max_cov = 10;      // 发散判定：方差太大说明完全测不准，放弃该点

// ------------------------------------------------------------------
// 重要的函数
/// 从 REMODE 数据集读取数据
// 修改后 (请直接替换为这一段)
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d, Eigen::aligned_allocator<SE3d>> &poses, 
    cv::Mat &ref_depth
);

/**
 * 根据新的图像更新深度估计
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         参考图像到当前图像的位姿
 * @param depth         深度
 * @param depth_cov     深度方差
 * @return              是否成功
 */
bool update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 极线搜索
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         位姿
 * @param pt_ref        参考图像中点的位置
 * @param depth_mu      深度均值
 * @param depth_cov     深度方差
 * @param pt_curr       当前点
 * @param epipolar_direction  极线方向
 * @return              是否成功
 */
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
);

/**
 * 更新深度滤波器
 * @param pt_ref    参考图像点
 * @param pt_curr   当前图像点
 * @param T_C_R     位姿
 * @param epipolar_direction 极线方向
 * @param depth     深度均值
 * @param depth_cov2    深度方向
 * @return          是否成功
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 计算 NCC 评分
 * @param ref       参考图像
 * @param curr      当前图像
 * @param pt_ref    参考点
 * @param pt_curr   当前点
 * @return          NCC评分
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// 双线性灰度插值
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------------------------
// 一些小工具
// 显示估计的深度图
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// 像素到相机坐标系
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相机坐标系到像素
inline Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 检测一个点是否在图像边框内
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

// 显示极线匹配
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// 显示极线
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

/// 评测深度估计
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


// ./dense_mapping ../../dense_mono/test_data
int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    

    // 从数据集读取数据
    vector<string> color_image_files;
    // vector<SE3d> poses_TWC;
    // 【C++知识点】Eigen::aligned_allocator
    // Eigen 的固定大小向量化类型（如 Vector3d, Matrix4d）需要 16 字节对齐。
    // 标准 STL vector 可能会破坏这种对齐，导致段错误。必须指定对齐分配器。
    vector<SE3d, Eigen::aligned_allocator<SE3d>> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // 读取第一帧作为“参考帧”(Reference Frame)
    Mat ref = imread(color_image_files[0], 0);       // 0 表示以灰度模式读取

    // 改动 2: 增加空指针检查
    if (ref.data == nullptr) {
        cout << "Fatal Error: Reference image is empty! Path: " << color_image_files[0] << endl;
        return -1;
    }

    SE3d pose_ref_TWC = poses_TWC[0];   // 以参考帧的位姿为世界位姿（坐标系） T_WC
    double init_depth = 3.0;    // 深度初始均值：假设所有像素都在 3米处
    double init_cov2 = 3.0;     // 深度初始方差：设得很大，表示非常不确定
    Mat depth(height, width, CV_64F, init_depth);             // 均值图
    Mat depth_cov2(height, width, CV_64F, init_cov2);         // 方差图
    /**
     * 深度滤波器原理: 我们把每个像素的深度看作一个高斯分布 $N(\mu, \sigma^2)$。初始时我们不知道深度是多少，
     * 所以给一个大概的均值 $\mu=3.0$，并给一个很大的方差 $\sigma^2=3.0$（表示很不确定）。随着观测增多，方差会越来越小（收敛）。
     */

    for (int index = 1; index < color_image_files.size(); index++) {
        // 读取当前帧 (Current Frame)
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;     // 忽略空指针
        SE3d pose_curr_TWC = poses_TWC[index];  // 获取当前帧的位姿

        // 【SLAM知识点】坐标系变换
        // 我们需要计算“参考帧”到“当前帧”的相对变换 T_C_R (T_Current_Reference)
        // 公式：T_C_R = T_C_W * T_W_R = T_WC^{-1} * T_WR       T_W_R（T_WR）表示参考帧到世界坐标系的变换
        // 这里的 poses_TWC 存储的是 T_W_C (世界坐标系到当前帧的变换)，T_W_C的逆，也就是T_C_W就是当前帧到世界坐标系的变换
        // pose_ref_TWC（也就是T_W_R） 存储的是世界坐标系到参考帧的变换矩阵
        // 简单来说，参考帧到当前帧的变换，就是参考帧到世界坐标系的变换矩阵 * 世界坐标系到当前帧的变换矩阵
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // 坐标转换关系： T_C_W * T_W_R = T_C_R

        // 调用核心更新函数
        update(ref, curr, pose_T_C_R, depth, depth_cov2);

        // 评测深度
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);

    cv::FileStorage fs("depth.xml", cv::FileStorage::WRITE);
    fs << "depth" << depth;
    fs.release();
    cout << "Depth map saved to depth.xml" << endl;


    // 将深度图归一化到 0~255 之间，这样原本 2.0 的值可能会变成 128 (灰色)，就能看清了
    cv::Mat depth_visual;
    // NORM_MINMAX 会自动把图中的最小值映射为0，最大值映射为255
    cv::normalize(depth, depth_visual, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("depth_visual.png", depth_visual);
    cout << " -> Saved visualization to 'depth_visual.png' (Normalized)" << endl;

    cout << "Done." << endl;
    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d, Eigen::aligned_allocator<SE3d>> &poses, // 这里也要改！
    cv::Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}

// 对整个深度图进行更新
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    // 【关键】加上这一行，开启多线程并行计算
    // 这会让 x 的循环被分配到不同的 CPU 核心上执行
    #pragma omp parallel for
    for (int x = boarder; x < width - boarder; x++){
        for (int y = boarder; y < height - boarder; y++) {
            // 遍历每个像素

            // 1. 检查是否需要更新：如果方差太小(已收敛)或太大(已发散)，则跳过
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) // 深度已收敛或发散
                continue;

            // 2. 极线搜索：试图在当前帧找到匹配点 pt_curr
            // 传入：当前深度均值 depth.ptr...[x] 和 标准差 sqrt(cov)
            Vector2d pt_curr;
            Vector2d epipolar_direction;
            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Vector2d(x, y),
                depth.ptr<double>(y)[x],
                sqrt(depth_cov2.ptr<double>(y)[x]),
                pt_curr,
                epipolar_direction
            );

            if (ret == false) // 匹配失败
                continue;

            // 取消该注释以显示匹配
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // 3. 深度融合：如果找到了匹配，用三角化计算新深度，并更新高斯分布
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
    }
    return true; // <--- 加上这一行！
}

// 极线搜索
// 方法见书 12.2 12.3 两节
bool epipolarSearch(
    const Mat &ref, const Mat &curr,
    const SE3d &T_C_R, const Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Vector2d &pt_curr, Vector2d &epipolar_direction) {
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;    // 参考帧的 P 向量

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // 按深度均值投影的像素
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));    // 按最小深度投影的像素
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));    // 按最大深度投影的像素

    Vector2d epipolar_line = px_max_curr - px_min_curr;    // 极线（线段形式）
    epipolar_direction = epipolar_line;        // 极线方向
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();    // 极线线段的半长度
    if (half_length > 100) half_length = 100;   // 我们不希望搜索太多东西

    // 取消此句注释以显示极线（线段）
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在极线上搜索，以深度均值点为中心，左右各取半长度
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;  // 待匹配点
        if (!inside(px_curr))
            continue;
        // 计算待匹配点与参考帧的 NCC
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)      // 只相信 NCC 很高的匹配
        return false;
    pt_curr = best_px_curr;
    return true;
}

double NCC(
    const Mat &ref, const Mat &curr,
    const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 计算 Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   // 防止分母出现零
}

// 这个函数利用对极几何约束，在当前帧的一条线上寻找与参考帧像素匹配的点
bool updateDepthFilter(const Vector2d &pt_ref,      // 参考帧的像素点
                        const Vector2d &pt_curr,    // 当前帧的像素点
                        const SE3d &T_C_R,          // 参考帧到当前帧的变换
                        const Vector2d &epipolar_direction,     // 极线方向
                        Mat &depth,                             // 深度均值
                        Mat &depth_cov2) {                      // 深度协方差
    // 不知道这段还有没有人看
    // 用三角化计算深度
    SE3d T_R_C = T_C_R.inverse();   // 当前帧到参考帧的变换

    // 将参考帧和当前帧的像素坐标转相机归一化坐标: P_cam = K^{-1} * p_pixel
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();  // 归一化为单位方向向量，参考帧射线方向
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();     // 当前帧射线方向

    // 三角化：已知同一个 3D 点在两张图上的投影方向（射线），求这两个射线的交点（即 3D 点的坐标）
    // 建立方程组，求解两帧下的深度 d_ref 和 d_cur
    // 实际上两条射线在 3D 空间几乎不可能相交（有噪声），所以求的是“公垂线中点”
    // 这里使用了克莱姆法则或直接求逆解 2x2 线性方程
    // 下式左右两侧都是表示的“同一个3D点的3维坐标”
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 由于噪声存在，两条射线在空间中通常无法完美相交（异面直线）。
    // 因此我们求公垂线的中点作为最优解（最小二乘解）转化成下面这个矩阵方程组
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]

    // 2. 构建 Ax=b 方程组
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;     // 旋转后的当前帧射线
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));     // 等式右边 [f_ref^T t, f2^T t]
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);

    // 3. 求解深度 d_ref (ans[0]) 和 d_cur (ans[1])
    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;           // ref 侧的结果
    Vector3d xn = t + ans[1] * f2;          // cur 结果
    Vector3d p_esti = (xm + xn) / 2.0;      // P的位置，取两者的平均
    double depth_estimation = p_esti.norm();   // 深度值

    // 计算不确定性（以一个像素为误差）
    // 如果匹配有一个像素的误差，算出来的深度会偏差多少？这个偏差就是这次观测的方差（不确定性）
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));

    // 给当前帧匹配的像素点模拟一个像素的误差
    // pt_curr + epipolar_direction 表示在极线上移动一个像素
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // 高斯融合
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

// 后面这些太简单我就不注释了（其实是因为懒）
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    double ave_depth_error = 0;     // 平均误差
    double ave_depth_error_sq = 0;      // 平方误差
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr) {

    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}
