#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/solvers/csparse/linear_solver_csparse.h>
// 由于我是用apt安装的系统级libg2o-dev，通常默认不包含CSparse扩展，大多数现代Linux发行版的G2O包主要支持Cholmod作为线性求解器
// 因为Chlomod性能通常更好且维护更活跃
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>  
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>
#include <memory>

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

/// 相机位姿和内参的结构
struct PoseAndIntrinsics {
    // 构造函数
    PoseAndIntrinsics() {}

    // set from given data address
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    // 将估计值放入内存
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    // 核心数据成员
    SO3d rotation;  // 旋转 (使用 Sophus::SO3d 李群)
    Vector3d translation = Vector3d::Zero();    // 平移 t
    double focal = 0;   // 焦距 f
    double k1 = 0, k2 = 0;  // 径向畸变系数
};

// 定义相机顶点（待优化参数包括相机位姿和内参）
// 继承 BaseVertex<9, PoseAndIntrinsics>
// 9: 优化变量的维度 (3旋转 + 3平移 + 1焦距 + 2畸变)
// PoseAndIntrinsics: 数据类型
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;    // Eigen 内存对齐宏，防止段错误

    VertexPoseAndIntrinsics() {}

    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();    // 重置为初始值
    }

    // 【核心】定义变量如何更新：x_new = x_old [+] delta_x
    virtual void oplusImpl(const double *update) override {
        // 1. 旋转更新：李群乘法 (左乘扰动模型)
        // R_new = exp(delta_phi) * R_old
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation; // update[0-2] 是旋转的李代数增量 (so3)
        
        // 2. 平移更新：向量加法
        _estimate.translation += Vector3d(update[3], update[4], update[5]);

        // 3. 相机焦距和径向畸变参数更新
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    // 根据估计值投影一个点，处理 BAL 数据集的特殊坐标系。
    // 将一个世界坐标系下的3D点先变换到相机坐标系，再投影到相机图像平面。
    Vector2d project(const Vector3d &point) {
        // 1. 世界系 -> 相机系: P_c = R * P_w + t
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        // 2. 归一化 (注意这里的负号，适配 BAL 数据集)
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

/**
 * 定义路标点顶点，待优化变量包括路标点的 xyz 3D坐标
 */
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    // 【核心】定义变量如何更新：x_new = x_old [+] delta_x
    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

/**
 * 定义观测边，每个相机和它拍摄到的每个路标点都会形成一条观测边
 * 继承自 BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint>
 * 2: 观测值的维度 (u, v) 
 * Vector2d: 观测值的数据类型
 * 两个连接的顶点类型：相机顶点, 路标顶点
 */
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 计算残差: error = prediction - measurement
    virtual void computeError() override {
        // 获取连接的两个顶点
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];     // 相机顶点
        auto v1 = (VertexPoint *) _vertices[1];                 // 路标点顶点

        // 计算投影
        auto proj = v0->project(v1->estimate());
        // 计算误差
        _error = proj - _measurement;
    }
    // 注意：这里没有覆盖 linearizeOplus 函数，意味着使用数值求导 (Numerical Derivative)
    // 如果为了速度，应该手写雅可比矩阵

    // use numeric derivatives
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

};

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

/**
 * 
 */
void SolveBA(BALProblem &bal_problem) {
    // 获取数据指针
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // 1. 定义 BlockSolver 的 Trait
    // <9, 3> 表示：Pose 维度是 9，Landmark 维度是 3
    // 这让 g2o 在编译期就能优化矩阵块的大小，极大提升速度
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;

    // 2. 选择线性求解器 (Linear Solver)
    // 这里选择了 Cholmod (稀疏 Cholesky 分解)，专治大规模稀疏矩阵
    // typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType;

    // 3. 选择优化算法 (LM 算法)
    // 组合关系：LM 算法 -> 包含 BlockSolver -> 包含 LinearSolver
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    /// build g2o problem
    const double *observations = bal_problem.observations();

    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));

        // 【关键！】设置边缘化 (Marginalized)
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);     // 迭代 40 次

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}
