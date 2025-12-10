/*
 * ======================================================================================
 * 文件名: plotTrajectory.cpp
 * 功能: 轨迹评估与可视化工具
 *
 * 描述:
 * 1. 读取两个轨迹文件（真实值 groundtruth 和 估计值 estimated）。
 * 文件格式通常为: [time, tx, ty, tz, qx, qy, qz, qw]
 * 2. 使用 Sophus::SE3d 存储每个位姿。
 * 3. 计算均方根误差 (RMSE - Root Mean Square Error)。
 * 利用李代数性质，计算 SE(3) 上的误差。
 * 4. 使用 Pangolin 库创建一个 GUI 窗口，将两条轨迹绘制在 3D 空间中进行对比。
 * - 蓝色线条: Ground Truth (真值)
 * - 红色线条: Estimated (估计值)
 *
 * 依赖库:
 * - Sophus: 用于处理李群 SE(3) 和李代数 se(3)
 * - Pangolin: 用于 OpenGL 3D 可视化及 GUI
 * - Eigen: 线性代数库 (Sophus 的基础)
 * ======================================================================================
 */

#include <iostream>
#include <fstream>
#include <unistd.h> // 用于 usleep 函数
#include <pangolin/pangolin.h> // 可视化库
#include <sophus/se3.hpp> // 李群库

using namespace Sophus;
using namespace std;

// 定义轨迹文件路径（硬编码路径，实际使用可能需要根据环境修改）
string groundtruth_file = "./example/groundtruth.txt";
string estimated_file = "./example/estimated.txt";

// 定义 TrajectoryType 为 Sophus::SE3d 的向量
// 注意：由于 SE3d 内部包含 Eigen 成员，使用 STL 容器时需要指定 Eigen::aligned_allocator 进行内存对齐
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

// 函数声明：绘制轨迹
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

// 函数声明：读取轨迹文件
TrajectoryType ReadTrajectory(const string &path);

int main(int argc, char **argv) {
    // 1. 读取数据
    TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
    TrajectoryType estimated = ReadTrajectory(estimated_file);

    // 检查读取是否成功以及数据量是否一致
    // 在实际评估中，通常需要根据时间戳对齐，这里假设数据已经是对应好的
    assert(!groundtruth.empty() && !estimated.empty());
    assert(groundtruth.size() == estimated.size());

    // 2. 计算 RMSE (均方根误差)
    double rmse = 0;
    for (size_t i = 0; i < estimated.size(); i++) {
        Sophus::SE3d p1 = estimated[i];   // 估计位姿 T_esti
        Sophus::SE3d p2 = groundtruth[i]; // 真实位姿 T_gt
        
        // --- 核心数学逻辑 ---
        // 计算误差变换矩阵: T_error = T_gt^{-1} * T_esti
        // p2.inverse() * p1 得到的是从真值坐标系到估计值坐标系的变换
        // .log() 将 SE(3) 元素映射到李代数 se(3) 空间 (生成一个 6维向量)
        // .norm() 计算这个 6维向量的 L2 范数
        double error = (p2.inverse() * p1).log().norm();
        
        rmse += error * error; // 累加平方误差
    }
    rmse = rmse / double(estimated.size()); // 计算均方
    rmse = sqrt(rmse); // 开根号得到 RMSE
    
    cout << "RMSE = " << rmse << endl;

    // 3. 绘制轨迹
    DrawTrajectory(groundtruth, estimated);
    
    return 0;
}

// 读取轨迹文件的具体实现
TrajectoryType ReadTrajectory(const string &path) {
    ifstream fin(path);
    TrajectoryType trajectory;
    if (!fin) {
        cerr << "trajectory " << path << " not found." << endl;
        return trajectory;
    }

    // 循环读取直到文件末尾
    while (!fin.eof()) {
        double time, tx, ty, tz, qx, qy, qz, qw;
        // 假设文件格式为：时间戳 tx ty tz qx qy qz qw
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        
        // 构建 SE3d 对象
        // 注意 Eigen::Quaterniond 的构造函数顺序是 (w, x, y, z)，而文件读取顺序通常是 xyzw
        // Sophus::SE3d 构造函数接受 (四元数, 平移向量)
        Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
        
        trajectory.push_back(p1);
    }
    return trajectory;
}

// 使用 Pangolin 绘制轨迹的具体实现
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
    // 创建一个名为 "Trajectory Viewer" 的窗口，大小 1024x768
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    
    // 启用深度测试（处理遮挡）
    glEnable(GL_DEPTH_TEST);
    // 启用混合（用于透明度等，虽然此处主要是画线）
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 定义相机的投影矩阵 (Projection Matrix) 和 观测矩阵 (ModelView Matrix)
    // Projection: 宽, 高, fx, fy, cx, cy, near, far
    // ModelViewLookAt: 相机位置(0, -0.1, -1.8), 看向的点(0, 0, 0), 上方向(0, -1, 0)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // 创建交互视图，并绑定上面的相机状态处理器
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // 渲染循环
    while (pangolin::ShouldQuit() == false) {
        // 清除颜色缓冲和深度缓冲
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 激活相机，准备绘制
        d_cam.Activate(s_cam);
        
        // 设置背景颜色为白色
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // 设置线宽
        glLineWidth(2);
        
        // --- 绘制 Ground Truth (蓝色) ---
        for (size_t i = 0; i < gt.size() - 1; i++) {
            glColor3f(0.0f, 0.0f, 1.0f); // 蓝色 RGB(0,0,1)
            glBegin(GL_LINES); // 开始画线模式
            auto p1 = gt[i], p2 = gt[i + 1];
            // 提取平移部分 (translation) 作为顶点坐标
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd(); // 结束画线
        }

        // --- 绘制 Estimated (红色) ---
        for (size_t i = 0; i < esti.size() - 1; i++) {
            glColor3f(1.0f, 0.0f, 0.0f); // 红色 RGB(1,0,0)
            glBegin(GL_LINES);
            auto p1 = esti[i], p2 = esti[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        
        // 交换帧缓冲区，完成渲染
        pangolin::FinishFrame();
        
        // 稍微休眠，减小 CPU 占用
        usleep(5000); // 5 ms
    }
}