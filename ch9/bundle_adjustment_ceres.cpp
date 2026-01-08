#include <iostream>
#include <ceres/ceres.h>
#include "common.h"     // 包含 BALProblem 类的定义（处理数据的读写）
#include "SnavelyReprojectionError.h"
#include <memory>
using namespace std;

void SolveBA(BALProblem &bal_problem);


// ./bundle_adjustment_ceres ../problem-16-22106-pre.txt
int main(int argc, char **argv) {
    // 1. 检查命令行参数：确保用户输入了数据文件路径
    if (argc != 2) {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
        return 1;
    }

    // 2. 加载数据
    // BALProblem 是书中封装好的类，负责解析那个复杂的 problem-16-22106-pre.txt 文件
    BALProblem bal_problem(argv[1]);
    // 3. 数据归一化
    // 将所有路标点的中心移动到原点，并进行缩放。这能显著提高优化的数值稳定性。
    bal_problem.Normalize();
    // 4. 添加噪声 (Perturb)
    // 这是一个测试环节。给原本较好的初值加上随机噪声，以此来验证算法是否能把"歪"了的结果优化回正确的。
    // 0.1 是旋转噪声，0.5 是平移噪声，0.5 是点坐标噪声
    bal_problem.Perturb(0.1, 0.5, 0.5);
    // 5. 保存初始状态点云 (用于可视化对比)
    bal_problem.WriteToPLYFile("initial.ply");
    // 6. 核心求解函数
    SolveBA(bal_problem);
    // 7. 保存优化后的点云
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

/**
 * 从数据到图优化模型（Graph）的构建
 */
void SolveBA(BALProblem &bal_problem) {
    // 获取参数块的大小：点是3维(x,y,z)，相机是9维(R,t,f,k1,k2)
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();

    // 获取指向数据数组首地址的指针
    // mutable_points() 返回可被修改的数组指针，优化器会直接修改这里面的值
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    // 获取观测数据数组：[u1, v1, u2, v2, ...]
    const double *observations = bal_problem.observations();
    ceres::Problem problem; // 实例化一个 Ceres 问题对象

    // 遍历所有的观测记录 (observations)，循环添加残差块（构建图的边）
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *cost_function;

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        // A. 创建代价函数
        // 利用 SnavelyReprojectionError::Create 工厂函数
        // 传入当前的观测值 u, v
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

        // B. 创建损失函数 (Huber Kernel)
        // 这是一个鲁棒核函数 (Robust Kernel)。
        // 作用：当误差非常大时（可能是外点 Outlier），Huber Loss 会降低其权重，
        // 防止个别错误的观测点把整个优化结果带偏。
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // C. 这里的指针运算非常关键！(Pointer Arithmetic)
        // 我们需要找到第 i 个观测对应的 "那个相机" 和 "那个路标点" 在大数组中的具体位置。
        
        // camera_index()[i] 是第 i 个观测对应的相机 ID
        // cameras 是首地址。每个相机占 camera_block_size (9) 个 double。
        // 所以地址 = 首地址 + 9 * ID
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        // 同理，找到对应的路标点地址
        // 地址 = 首地址 + ID * 3
        double *point = points + point_block_size * bal_problem.point_index()[i];

        // D. 添加残差块 (AddResidualBlock)
        // 这相当于在图优化中添加了一条"边"。
        // 边连接了两个节点：camera 和 point。
        // 边的约束由 cost_function 定义。
        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;

    // 配置求解器选项
    ceres::Solver::Options options;
    // A. 这里的设置至关重要：SPARSE_SCHUR
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    // 设置为 true 以便在控制台看到每一轮迭代的误差下降情况
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    // B. 开始求解 (Solve)
    // 这是一个阻塞函数，对于大问题可能需要跑几秒甚至几分钟
    ceres::Solve(options, &problem, &summary);

    // 输出最终报告
    std::cout << summary.FullReport() << "\n";
}