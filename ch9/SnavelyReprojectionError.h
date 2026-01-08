#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"   // 一个辅助文件，包含罗德里格斯公式的实现

/**
 * 这份代码定义了一个名为 SnavelyReprojectionError 的类，它是 Ceres Solver 用来计算 重投影误差（Reprojection Error） 的核心组件。
 * 
 * 在 Bundle Adjustment (BA) 中，我们的目标是最小化这个误差。
 * 简单来说，这个类的作用是告诉 Ceres：“给定当前的相机参数和空间点坐标，算出来的像素位置（预测值）和真实拍到的像素位置（观测值）差了多少？”
 */
class SnavelyReprojectionError {
public:
    // 构造函数：传入真实的观测值 (u, v)
    // observed_x 和 observed_y 是我们在照片上提取到的特征点坐标（也就是我们在 txt 文件里读到的那两个数）
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                           observed_y(observation_y) {}


    /**
     * 这是一个仿函数类，在C++中，如果一个类重载了operator()，它就可以像函数一样被调用。
     * Ceres 利用这种机制来计算残差
     * 
     * 使用template<typename T>模板，是为例支持Ceres的自动微分，当Ceres求解时，T会被替换为ceres::Jet类型（包含数值和导数的复数结构）
     * 通过这种方式，ceres可以在运行代码计算残差的同时，利用链式法则自动算出雅可比矩阵，就不用手推偏导数，计算数值微分了。
     */
    template<typename T>
    bool operator()(const T *const camera,  // 9维相机参数数组
                    const T *const point,   // 3维路标点坐标数组
                    T *residuals) const {   // 输出：2维残差
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];   // predictions 数组用于存储计算出来的预测像素坐标
        CamProjectionWithDistortion(camera, point, predictions);    // 调用具体的投影函数（考虑径向畸变）
        
        // 计算残差 = 预测值 - 观测值
        residuals[0] = predictions[0] - T(observed_x);  // 这里将double类型的观测值强制转换为模板类型T，以保证类型一致
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) {
        // 1. 旋转：使用罗德里格斯公式 (Rodrigues' formula)
        // camera[0,1,2] 是旋转向量 (Angle-Axis)
        T p[3];
        AngleAxisRotatePoint(camera, point, p);

        // 2. 平移
        // camera[3,4,5] 是平移向量 t
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // 3. 计算归一化坐标，并处理 BAL 数据集的特殊投影方向
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion
        const T &l1 = camera[7];    // 二阶畸变系数 k1
        const T &l2 = camera[8];    // 四阶畸变系数 k2

        // 计算半径的平方 r^2 = x^2 + y^2
        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);    // 这里只考虑了径向畸变（物体离中心越远，变形越大），忽略了切向畸变;代码为了效率，提取了公因式 r2

        // 5. 应用焦距 f，转换到像素坐标
        // 这里隐含假设了光心主点 $(c_x, c_y)$ 是图像中心（即为0，或者数据已经预处理减去了中心），且 $f_x = f_y$（正方形像素）。
        const T &focal = camera[6];
        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        // 创建自动微分 CostFunction
        // <SnavelyReprojectionError, 2, 9, 3> 是模板参数
        /**
         * AutoDiffCostFunction 是 Ceres 最强大的功能之一。
         * 模板参数 <SnavelyReprojectionError, 2, 9, 3> 的含义极其重要：
         *      SnavelyReprojectionError：使用的仿函数类型。
         *      2：残差的维数（输出维度）。因为是图像上的 $(u, v)$ 误差，所以是2。
         *      9：第一个参数块的维数。即相机参数（旋转3 + 平移3 + 焦距1 + 畸变2 = 9）。
         *      3：第二个参数块的维数。即路标点坐标（X, Y, Z）。
        */
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h

