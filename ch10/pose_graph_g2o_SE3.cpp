/*
 * 项目名称: 基于g2o的SE3位姿图优化 (Pose Graph Optimization)
 * 功能描述: 
 * 1. 读取 g2o 格式的文本数据文件 (sphere.g2o)。
 * 2. 手动解析文件中的顶点 (Vertex) 和边 (Edge) 信息。
 * 3. 构建非线性最小二乘优化问题 (图优化模型)。
 * 4. 使用 Levenberg-Marquardt 算法优化机器人的轨迹位姿。
 * 5. 将优化后的结果保存为新的 g2o 文件。
 *
 * 核心知识点:
 * - SLAM 后端: 位姿图优化 (只有位姿，没有路标点)。
 * - 数学基础: 李群 SE(3) 与李代数 se(3)，四元数，非线性最小二乘。
 * - 编程技巧: C++ 文件流操作，g2o 库的 BlockSolver 和 OptimizationAlgorithm 配置。
 *
 * 输入: sphere.g2o (包含带噪声的球形轨迹数据)
 * 输出: result.g2o (优化后的轨迹数据)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <memory>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是四元数而非李代数.
 * **********************************************/

 // ./pose_graph_g2o_SE3 ../sphere.g2o
int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // 2. 配置 g2o 求解器 (核心步骤)
    // 定义块求解器的维度特征：<6, 6> 表示位姿(Pose)是6维，观测(Edge)也是6维
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    // 定义线性求解器：使用 Eigen 的稀疏 Cholesky 分解
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

    // 初始化 LM 算法 (非线性优化算法)，使用 unique_ptr 智能指针管理内存
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 定义稀疏优化器（图模型对象）
    optimizer.setAlgorithm(solver);     // 设置求解算法
    optimizer.setVerbose(true);         // 开启调试输出，可以看到迭代过程中的误差变化

    // 3. 读取数据并构建图
    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    while (!fin.eof()) {
        string name;
        fin >> name;    // 读取行首的标签
        if (name == "VERTEX_SE3:QUAT") {
            // --- 处理顶点 (机器人位姿) ---
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);       // g2o内置的解析函数，读取 x,y,z,qx,qy,qz,qw
            optimizer.addVertex(v);
            vertexCnt++;

            // SLAM数学原理：固定第一个顶点，消除规范自由度 (Gauge Freedom)
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // --- 处理边 (相对位姿约束) ---
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1, idx2;     // 关联的两个顶点
            fin >> idx1 >> idx2;    // 读取该边连接的两个顶点ID
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);    // 连接第一个点
            e->setVertex(1, optimizer.vertices()[idx2]);    // 连接第二个点
            e->read(fin);   // 读取观测值(相对位姿)和信息矩阵(协方差的逆)
            optimizer.addEdge(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    // 4. 执行优化
    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();     // 初始化优化
    optimizer.optimize(30);                 // 迭代30次

    // 5. 保存结果
    cout << "saving optimization results ..." << endl;
    optimizer.save("result.g2o");

    return 0;
}