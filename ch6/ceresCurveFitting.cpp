//
// Created by xiang on 18-11-19.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>	// 核心库，Google 的 Ceres Solver，用于非线性优化。
#include <chrono>

using namespace std;

// 代价函数的计算模型
struct CURVE_FITTING_COST{
	// 构造函数：每产生一个观测数据，就构建一个这样的结构体
	CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

	// 残差的计算
	// 这是一个模板函数，因为 Ceres 需要支持自动求导 (Auto Diff)
    // T 可能是 double，也可能是 Ceres 内部定义的 Jet 类型（用于携带导数信息）
	template <typename T>
	bool operator()(
		const T *const abc, // 模型参数，有3维
		T *residual) const {	// 残差，输出：计算出来的误差

		// 公式：residual = y - exp(ax^2 + bx + c)
        // 注意：这里必须使用 ceres::exp 而不是 std::exp，以支持自动求导
        // T(_x) 将 double 类型的数据强制转换为 T 类型，保证类型一致
		residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
		return true;
	}

	const double _x, _y; 	// 存储观测数据 x, y
};

int main(int argc, char **argv){
	double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
	double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
	int N = 100;						  // 数据点
	double w_sigma = 1.0;				  // 噪声Sigma值
	double inv_sigma = 1.0 / w_sigma;
	cv::RNG rng; 		// OpenCV随机数产生器

	vector<double> x_data, y_data; // 数据

	// 生成观测数据：真实模型值 + 高斯噪声
	for (int i = 0; i < N; i++){
		double x = i / 100.0;
		x_data.push_back(x);
		y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
	}

	// 待估计参数，用数组存储，初始值为估计值 (2.0, -1.0, 5.0)
	double abc[3] = {ae, be, ce};

	// 构建最小二乘问题
	ceres::Problem problem;		// 声明一个优化问题对象
	for (int i = 0; i < N; i++){
		// AddResidualBlock 向问题中添加误差项（在 g2o 中叫“边”）
		// 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
		
		problem.AddResidualBlock( 
			// 1. 使用自动求导 (AutoDiffCostFunction)
			// <CURVE_FITTING_COST, 1, 3>:
			//    - CURVE_FITTING_COST: 我们定义的仿函数类型
			//    - 1: 残差的维度 (输出维度)，这里是一维标量
			//    - 3: 优化变量的维度 (输入维度)，即 a, b, c 共3个
			new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])),

			// 2. 核函数 (Loss Function)
            // nullptr 表示使用标准的最小二乘误差 (L2 Loss)。
            // 如果有离群点，可以使用 new ceres::HuberLoss(1.0) 等来抑制异常值。
			nullptr, // 核函数，这里不使用，为空

			// 3. 待优化的参数块 (Parameter Block)
            // 这里传入的是数组的首地址。Ceres 会直接修改这个内存里的值。
			abc		 // 待估计参数
		);
	}

	// 配置求解器
	ceres::Solver::Options options;							   // 这里有很多配置项可以填
	// 选择线性求解器类型：DENSE_NORMAL_CHOLESKY
    // 因为这是个小规模问题，Hessian 矩阵是稠密的 (3x3)，用 Cholesky 分解求解 Hx=g 非常快。
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // 增量方程如何求解

	// 允许输出迭代过程到标准输出 (终端)
	options.minimizer_progress_to_stdout = true;			   // 输出到cout

	ceres::Solver::Summary summary; 	// 定义一个对象来存储优化结果报告
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	ceres::Solve(options, &problem, &summary); // 开始优化
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

	// 输出结果
	cout << summary.BriefReport() << endl;
	cout << "estimated a,b,c = ";
	for (auto a : abc)
		cout << a << " ";
	cout << endl;

	return 0;
}