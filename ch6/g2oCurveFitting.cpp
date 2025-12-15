#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>		// 顶点基类
#include <g2o/core/base_unary_edge.h>	// 一元边基类
#include <g2o/core/block_solver.h>		// 块求解器
#include <g2o/core/optimization_algorithm_levenberg.h>		// Levenberg-Marquardt 迭代法
#include <g2o/core/optimization_algorithm_gauss_newton.h>	// Gauss-Newton 迭代法
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>		// 稠密线性求解器
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
#include <memory>		// 智能指针
using namespace std;

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
// 继承自 g2o::BaseVertex
// 模板参数 <3, Eigen::Vector3d>:
// 3: 优化变量的维度 (a, b, c 共3个)
// Eigen::Vector3d: 优化变量的数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW		// Eigen 内存对齐宏，防止内存错误

	// 重置
	virtual void setToOriginImpl() override
	{
		_estimate << 0, 0, 0;		// 把估计值重置为0
	}

	// 更新函数：定义如何把增量 (update) 加到当前估计值 (_estimate) 上
    // 这里的 update 是由优化算法计算出来的 delta_x
	virtual void oplusImpl(const double *update) override
	{
		_estimate += Eigen::Vector3d(update);
	}

	// 存盘和读盘：留空
	// 读写函数：g2o 要求实现，但如果不用文件读写图结构，可以直接返回 true
	virtual bool read(istream &in) {return true;}

	virtual bool write(ostream &out) const {return true;}
};


// 误差模型 模板参数：观测值维度，类型，连接顶点类型
// 继承自 g2o::BaseUnaryEdge
// 模板参数 <1, double, CurveFittingVertex>:
// 1: 观测值的维度 (y 值是1维的)
// double: 观测值的数据类型
// CurveFittingVertex: 连接的顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// 构造函数：传入 x 值
	// 误差模型 y = exp(ax^2 + bx + c)
	CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

	// 计算曲线模型误差
	// 误差 = 观测值 - 预测值
	virtual void computeError() override
	{
		// 1. 获取连接的顶点 (也就是当前的 a, b, c 估计值)
		const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
		const Eigen::Vector3d abc = v->estimate();

		// 2. 计算误差：_measurement 是 y_data[i]，后面是模型公式 exp(ax^2+bx+c)
		_error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
	}

	// 计算雅可比矩阵
	// 误差对优化变量 (a, b, c) 的偏导数
	virtual void linearizeOplus() override
	{
		const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
		const Eigen::Vector3d abc = v->estimate();
		double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
		_jacobianOplusXi[0] = -_x * _x * y;    // 对a的偏导数
		_jacobianOplusXi[1] = -_x * y;		   // 对b的偏导数
		_jacobianOplusXi[2] = -y;			   // 对c的偏导数
	}

	/**
	 * 读取输入流中的数据（默认实现）
	 * 
	 * 该虚函数用于从输入流中读取数据，子类可重写以提供具体实现。
	 * 默认实现直接返回 true，表示读取操作成功，但实际未进行任何数据读取。
	 * 
	 * @param in 输入流对象
	 * @return 始终返回 true
	 */
	virtual bool read(istream &in) { return true; } // <--- 加上 return true;

	/**
	 * 写入输出流中的数据（默认实现）
	 * 
	 * 该虚函数用于将数据写入输出流，子类可重写以提供具体实现。
	 * 默认实现直接返回 true，表示写入操作成功，但实际未进行任何数据写入。
	 * 
	 * @param out 输出流对象
	 * @return 始终返回 true
	 */
	virtual bool write(ostream &out) const { return true; } // <--- 加上 return true;

public:
	double _x; // 存储观测数据中的 x 值， y 值为 _measurement
};

int main(int argc, char **argv)
{
	double ar = 1.0, br = 2.0, cr = 1.0;  	// 真实参数值
	double ae = 2.0, be = -1.0, ce = 5.0; 	// 估计参数值，故意设置得离真实值有偏差，看优化能否拉回来
	int N = 100;						  	// 100个数据点
	double w_sigma = 1.0;				  	// 噪声Sigma值
	double inv_sigma = 1.0 / w_sigma;
	cv::RNG rng; 	// OpenCV随机数产生器

	vector<double> x_data, y_data; // 数据
	for (int i = 0; i < N; i++)
	{
		double x = i / 100.0;
		x_data.push_back(x);
		// 生成带噪声的观测数据 y
		y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
	}

	// 构建图优化，先设定g2o
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;			  // 每个误差项优化变量维度为3，误差值维度为1
	typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 稠密线性求解器类型，适合这类小规模问题

	// 梯度下降方法，可以从GN, LM, DogLeg 中选
	// 3. 创建总求解器 (Solver)
    // 这是一个三层嵌套结构：
    // 算法(高斯牛顿) -> 块求解器(管理矩阵块) -> 线性求解器(解方程 Hx=-b)
    // 使用 std::make_unique 创建智能指针，避免内存泄漏，这是现代 C++ 写法
	auto solver = new g2o::OptimizationAlgorithmGaussNewton(
		std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

	// 4. 创建稀疏优化器 (Optimizer) 并设置算法
	g2o::SparseOptimizer optimizer; // 图模型
	optimizer.setAlgorithm(solver); // 设置求解器
	optimizer.setVerbose(true);		// 开启调试输出，可以看到每次迭代的 chi2 误差

	// 往图中增加顶点
	CurveFittingVertex *v = new CurveFittingVertex();
	v->setEstimate(Eigen::Vector3d(ae, be, ce));		// 设置优化的初始估计值 (2.0, -1.0, 5.0)
	v->setId(0);		// 设置顶点 ID
	optimizer.addVertex(v);

	// 往图中增加边
	for (int i = 0; i < N; i++)
	{
		CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);		// 传入 x
		edge->setId(i);
		edge->setVertex(0, v);																	 // 设置连接的顶点
		edge->setMeasurement(y_data[i]);														 // 观测数值
		// 设置信息矩阵 (Information Matrix)
        // 信息矩阵 = 协方差矩阵的逆。这里协方差是 sigma^2 * I，所以信息矩阵是 (1/sigma^2) * I
        // 信息矩阵代表了对这条数据的“信任程度”，噪声越大，信任度越低
		edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // 信息矩阵：协方差矩阵之逆
		optimizer.addEdge(edge);
	}

	// 执行优化
	cout << "start optimization" << endl;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();		// 计时开始
	optimizer.initializeOptimization();		// 初始化：检查图结构，分配内存
	optimizer.optimize(300);		// 迭代优化 10 次
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();		// 计时结束
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

	// 输出优化值
	Eigen::Vector3d abc_estimate = v->estimate();		// 获取优化后的最终值
	cout << "estimated model: " << abc_estimate.transpose() << endl;

	return 0;
}