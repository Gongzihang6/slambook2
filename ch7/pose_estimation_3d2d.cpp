#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);


// ./pose_estimation_3d2d ../1.png ../2.png ../1_depth.png ../2_depth.png
int main(int argc, char **argv){
    if (argc != 5){
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);      // 获取图1和图2彩色图关键点匹配关系
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat d1 = imread(argv[3], cv::IMREAD_UNCHANGED);     // 相机1位姿下的深度图，深度图为16位无符号数，单通道图像
    // 相机内参矩阵 K (fx, 0, cx; 0, fy, cy; 0, 0, 1)
    // 这里的数据是TUM数据集相机的典型参数
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;     // 存储图1对应的3D点
    vector<Point2f> pts_2d;     // 存储图2对应的2D像素点
    // 根据彩色图关键点匹配点对，从深度图1中获取匹配关键点的3D点
    for (DMatch m : matches){
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)     // bad depth
            continue;
        float dd = d / 5000.0;  // 将深度单位从毫米转换为米 (TUM数据集的缩放因子是5000)
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);      // 从像素坐标到相机归一化坐标
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));        // 存储相机1坐标系下的3D点，乘以深度dd，从相机归一化坐标到相机坐标系下的3D点坐标
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);               // 存储图2对应的2D像素点
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    // 调用OpenCV的PnP求解。
    // r 为旋转向量，t 为平移向量。
    // false 表示不使用初值 guess。
    // 默认算法通常是 EPNP 或 Iterative 方法。
    // 将求解出旋转和平移赋值给r，t
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues(r, R);    // r为旋转向量形式，用Rodrigues公式转换为矩阵
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

    cout << "R=" << endl
         << R << endl;
    cout << "t=" << endl
         << t << endl;

    
    // -- 5. 数据转换：OpenCV -> Eigen --
    // 为了后续的手写优化和g2o，需要将数据转换为Eigen格式
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i){
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    // -- 6. 方法二：手写高斯牛顿法 (Gauss-Newton) --
    cout << "calling bundle adjustment by gauss newton" << endl;
    Sophus::SE3d pose_gn;   // 初始化位姿 pose_gn，通常设为单位阵或给一个初始猜测
    t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

    // -- 7. 方法三：g2o 图优化 --
    cout << "calling bundle adjustment by g2o" << endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
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


/**
 * @brief 从像素坐标到相机归一化平面坐标
 * @param p 输入像素点
 * @param K 相机内参
 * @return 相机归一化平面坐标
 * 公式：
 * x = (u-cx)/fx
 * y = (v-cy)/fy
 */
Point2d pixel2cam(const Point2d &p, const Mat &K){
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,
                                const VecVector2d &points_2d,
                                const Mat &K,
                                Sophus::SE3d &pose){
    typedef Eigen::Matrix<double, 6, 1> Vector6d;       // 定义6维向量（对应se3李代数：3平移+3旋转）
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    // 从内参矩阵提取 fx, fy, cx, cy
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++){
        // 高斯牛顿法的核心方程：H * dx = g (或者 H * dx = -b)
        // H = J^T * J (海森矩阵的近似)
        // b = -J^T * e (负梯度)
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        // 遍历每一个点对，累加 H 和 b
        for (int i = 0; i < points_3d.size(); i++){
            // 1. 将3D点变换到当前相机坐标系下： P' = T * P
            Eigen::Vector3d pc = pose * points_3d[i];

            // 2. 投影到归一化平面并应用内参（为了计算误差）
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            // 预测的像素坐标 proj (u_est, v_est)
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
            
            // 3. 计算重投影误差 e = 观测值 - 预测值
            Eigen::Vector2d e = points_2d[i] - proj;

            // 累加Cost (误差平方和)
            cost += e.squaredNorm();

            // 4. 计算雅可比矩阵 J (2x6)
            // J 是 误差e 对 李代数扰动δξ 的导数: de/dδξ
            // 根据链式法则： de/dδξ = -(de/dP') * (dP'/dδξ)
            // 书中推导出的公式如下 (注意代码中因为 e = measured - projected，所以有一个负号的区别，或者直接用 projected - measured)
            // 这里的 J 是对应《十四讲》P166页 公式7.46的实现
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;

            // 5. 累加 H 和 b
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }
        // 6. 求解线性方程 H * dx = b
        // 使用LDLT分解求解比求逆更稳定
        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])){
            cout << "result is nan!" << endl;
            break;
        }
        // 7. 检查Cost是否下降，如果上升说明步长太大或线性化近似不够好（实际复杂的BA会用LM法调整lambda）
        if (iter > 0 && cost >= lastCost){
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6){
            // converge
            break;
        }
    }

    cout << "pose by g-n: \n"
         << pose.matrix() << endl;
}

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override{
        _estimate = Sophus::SE3d();
    }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override{
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

    virtual void computeError() override{
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override{
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi
            << -fx / Z,
            0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
            0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(istream &in) override { return true; }

    virtual bool write(ostream &out) const override { return true; }

private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose){

    // -- 1. 配置 g2o 优化器 --
    
    // 定义BlockSolver类型：
    // BlockSolverTraits<6, 3> 表示：
    // - 优化的变量（Pose）维度是 6
    // - 路标点（Landmark）维度是 3 (虽然这里路标点是固定的，edge只连了一个Pose，但Traits通常这样写)
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;           // pose is 6, landmark is 3
    // 线性求解器：使用稠密Cholesky分解 (LinearSolverDense)，因为只有Pose一个变量，H矩阵很小且稠密
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    // 选择优化算法：高斯牛顿法 (GaussNewton)
    // 也就是利用 H * dx = -b 进行迭代
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // 创建稀疏优化器（图模型）
    optimizer.setAlgorithm(solver); // 设置求解算法
    optimizer.setVerbose(true);     // 打开调试输出

    // -- 2. 添加顶点 (Vertex) --
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());   // 初始估计设为单位阵
    optimizer.addVertex(vertex_pose);

    // -- 3. 添加边 (Edge) --
    // 将Mat类型的K转换为Eigen类型
    Eigen::Matrix3d K_eigen;
    K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i){
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        // 创建边，绑定3D点和K
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);    // 连接到id为0的顶点(Camera Pose)
        edge->setMeasurement(p2d);          // 设置观测值(像素坐标)
        edge->setInformation(Eigen::Matrix2d::Identity());  // 信息矩阵(协方差逆)，这里设为单位阵表示所有点权重一致
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    // -- 4. 开始优化 --
    optimizer.initializeOptimization();
    optimizer.optimize(10);     // 迭代10次
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    // -- 5. 获取结果 --
    // 从顶点中取出估计好的 SE3
    cout << "pose estimated by g2o =\n"
         << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate();
}
