/**
 * @file sophus_demo.cpp
 * @brief 演示 Sophus 库中 SO(3) 与 SE(3) 李群/李代数基本用法的示例程序
 *
 * 本程序涵盖：
 * 1. 从旋转矩阵、四元数构造 SO(3) 李群
 * 2. 李群 ↔ 李代数（so3）的指数/对数映射
 * 3. 向量 ↔ 反对称矩阵的 hat/vee 操作
 * 4. 李代数增量扰动模型的更新
 * 5. 从旋转+平移构造 SE(3) 李群
 * 6. SE(3) 李群 ↔ 李代数（se3）的指数/对数映射及其更新
 *
 * 依赖：
 * - Eigen3
 * - Sophus
 *
 */

# include <iostream>
# include <cmath>
# include <Eigen/Core>
# include <Eigen/Geometry>
# include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

/// 本程序演示sophus的基本用法
int main(int argc, char **argv) {
	// 沿Z轴转90度的旋转矩阵
	Matrix3d R = AngleAxisd(M_PI/2,Vector3d(0,sqrt(2)/2,sqrt(2)/2)).toRotationMatrix();
	// 或者四元数
	Quaterniond q(R);		// 从旋转矩阵构造四元数
	Sophus::SO3d SO3_R(R);  // Sophus::SO3d（李群）可以直接从旋转矩阵构造
	Sophus::SO3d SO3_q(q);  // 也可以通过四元数构造
	// 二者是等价的
	cout << "SO(3) from matrix: \n" << SO3_R.matrix() << endl;
	cout << "SO(3) from quaternion: \n" << SO3_q.matrix() << endl;
	cout << "they are equal" << endl;

	// 使用对数映射获得它的李代数
	Vector3d so3 = SO3_R.log();
	cout << "so3 = " << so3.transpose() << endl;
	// hat为向量到反对称矩阵
	cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << endl;
	// 相对的，vee为反对称到向量
	cout << "so3 hat vee= " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

	// 增量扰动模型的更新
	Vector3d update_so3(1e-4, 0, 0); // 假设李代数更新量为这么多
	Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;	// 李代数到李群的指数映射，计算更新后的李群
	cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

	cout << "**********************变换矩阵的利群与李代数转化与扰动***************" << endl;
	// 对SE(3)操作大同小异
	Vector3d t(1, 0, 0); // 沿X轴平移1
	Sophus::SE3d SE3_Rt(R, t); // 从R,t构造SE(3)
	Sophus::SE3d SE3_qt(q, t); // 从q,t构造SE(3)
	cout << "SE3 from R,t= \n" << SE3_Rt.matrix() << endl;
	cout << "SE3 from q,t= \n" << SE3_qt.matrix() << endl;
	// 李代数se(3)是一个六维向量，方便起见先typedef一下
	typedef Eigen::Matrix<double, 6, 1> Vector6d;	// 定义变换矩阵的李代数（一个6维向量）数据类型
	Vector6d se3 = SE3_Rt.log();	// 利群到李代数的映射，计算变换矩阵的李代数表示
	cout << "se3 = " << se3.transpose() << endl;
	// 观察输出，会发现在Sophus中，se(3)的“平移在前，旋转在后”。
	// 同样地，有hat和vee两个算符
	cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
	cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

	// 最后，演示更新
	Vector6d update_se3; // 更新量
	update_se3.setZero();
	update_se3(0) = 1e-4;
	Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
	cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;
	return 0;
}
/*
SO(3) from matrix: 
        0 -0.707107  0.707107
 0.707107       0.5       0.5
-0.707107       0.5       0.5
SO(3) from quaternion: 
        0 -0.707107  0.707107
 0.707107       0.5       0.5
-0.707107       0.5       0.5
they are equal
so3 =       0 1.11072 1.11072
so3 hat=
       0 -1.11072  1.11072
 1.11072        0       -0
-1.11072        0        0
so3 hat vee=       0 1.11072 1.11072
SO3 updated = 
        0 -0.707107  0.707107
 0.707177   0.49995   0.49995
-0.707036   0.50005   0.50005
********************
SE3 from R,t= 
        0 -0.707107  0.707107         1
 0.707107       0.5       0.5         0
-0.707107       0.5       0.5         0
        0         0         0         1
SE3 from q,t= 
        0 -0.707107  0.707107         1
 0.707107       0.5       0.5         0
-0.707107       0.5       0.5         0
        0         0         0         1
se3 = 0.785398 -0.55536  0.55536        0  1.11072  1.11072
se3 hat = 
       0 -1.11072  1.11072 0.785398
 1.11072        0       -0 -0.55536
-1.11072        0        0  0.55536
       0        0        0        0
se3 hat vee = 0.785398 -0.55536  0.55536        0  1.11072  1.11072
SE3 updated = 
        0 -0.707107  0.707107    1.0001
 0.707107       0.5       0.5         0
-0.707107       0.5       0.5         0
        0         0         0         1
*/