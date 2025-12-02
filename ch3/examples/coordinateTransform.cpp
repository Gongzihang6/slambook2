/**
 * @file coordinateTransform.cpp
 * @brief 演示如何使用 Eigen 进行三维空间中的坐标变换
 *
 * 本例程展示了：
 * 1. 使用四元数表示旋转，并将其转换为 Isometry3d（齐次变换矩阵）
 * 2. 构造从世界坐标系 W 到机器人 1 号、2 号坐标系的变换 T1w、T2w
 * 3. 计算并输出点 p1 在机器人 1 号坐标系中的坐标，经 T2w * T1w^{-1} 变换后，在机器人 2 号坐标系中的坐标 p2
 *
 * 运行结果将打印 p2 的坐标，帮助理解三维空间中的刚体变换链式法则。
 */


#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
	// 用四元数来表示小罗卜1号和2号的坐标系相对于世界坐标系W的旋转关系
	Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
	q1.normalize();
	q2.normalize();
	// t1 表示机器人 1 原点在世界坐标系中的位置（位移）
	// t2 表示机器人 2 原点在世界坐标系中的位置（位移）
	Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);
	// p1 表示机器人 1 中某个点的坐标（在机器人 1 坐标系中）
	Vector3d p1(0.5, 0, 0.2);

	Isometry3d T1w(q1), T2w(q2);
	T1w.pretranslate(t1);	// 表示从世界坐标系W到机器人1号坐标系的变换
	T2w.pretranslate(t2);	// 表示从世界坐标系W到机器人2号坐标系的变换

	// 表示从机器人1号坐标系到机器人2号坐标系的变换
	Vector3d p2 = T2w * T1w.inverse() * p1;		// 机器人1坐标系 --> 世界坐标系 --> 机器人2坐标系
	cout << endl << p2.transpose() << endl;
	return 0;
}