#pragma once

#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "myslam/common_include.h"

namespace myslam {

// Pinhole stereo camera model
class Camera {
public:
    /**
     * 类成员变量中使用了 Eigen 的类型（如 SE3, Vec3, Mat33）。
     * 在 64 位系统中，Eigen 为了利用 SIMD（单指令多数据）指令集加速运算，要求内存首地址必须是 16 字节对齐的。
     * 
     * 后果：如果不加这个宏，使用 new Camera() 分配内存时，
     * C++ 默认的内存分配器可能不会按照 16 字节对齐，导致程序运行时直接崩溃（Segfault）。这个宏重载了 new 和 delete 操作符来保证对齐。
     */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * 相机参数通常是全局唯一的、共享的资源。前端需要查参数，后端优化需要查参数，可视化也需要。
     * 使用 shared_ptr 可以安全地在各个模块间共享同一个相机对象，而不用频繁拷贝，也不用担心谁负责释放内存。
     */
    typedef std::shared_ptr<Camera> Ptr;

    /**
     * 静态参数与动态状态分离，这里的 Camera 类存储的是静态参数（硬件出厂就定了，运行过程中一般不变）
     *      fx, fy, cx, cy: 针孔相机模型的 4 个核心内参。
     *      baseline: 双目相机的基线（左右眼距离）。如果是单目，这里就是 0。
     * pose_ (外参)：这是一个极其重要的概念。
     *      这里的 pose_ 不是 相机在世界坐标系中的位置（那是 Frame 类里的 T_wc）
     *      这里的 pose_ 是 安装外参。比如机器人身上装了两个相机，左眼是主参考系，那么左眼的 pose_ 通常是单位阵（Identity），
     *    而右眼的 pose_ 包含了一个平移向量（基线）
     * 设计目的：支持多相机系统。如果有 4 个相机，每个相机都有一个相对于“车体中心”或“主相机”的固定变换。
     */
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0,
           baseline_ = 0;  // Camera intrinsics
    SE3 pose_;             // extrinsic, from stereo camera to single camera
    SE3 pose_inv_;         // inverse of extrinsics


    /**
     * 在构造函数中预先计算 pose_inv_
     * 因为求逆矩阵是一个耗时操作。如果在实时追踪过程中（每秒 30 帧，每帧几百个点）每次都要现场求逆，会浪费大量 CPU。
     * 空间换时间是 SLAM 优化的常用手段。
     */
    Camera();
    Camera(double fx, double fy, double cx, double cy, double baseline,
           const SE3 &pose)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {
        pose_inv_ = pose_.inverse();
    }

    SE3 pose() const { return pose_; }

    // return intrinsic matrix
    Mat33 K() const {
        Mat33 k;
        k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return k;
    }

    // coordinate transform: world, camera, pixel
    /**
     * 这部分定义了 SLAM 中最频繁使用的三大坐标系转换：
     *      World (世界坐标系): $P_w$
     *      Camera (相机坐标系): $P_c$ 
     *      Pixel (像素坐标系): $P_{uv}$
     * 在 SLAM 系统中，Camera 类充当了 “几何服务层” (Geometry Service Layer) 的角色：
     *  抽象化 (Abstraction)：
     *      前端（Frontend）和后端（Backend）不需要关心投影的具体数学公式是针孔模型还是鱼眼模型。它们只需要调用 camera->world2pixel(pt)。
     *      如果未来你要换成鱼眼相机，只需要修改 Camera 类的实现，而不用去改前端追踪的一行代码
     *  解耦 (Decoupling)：
     *      将**“相机的物理属性”（Camera类）与“相机的运动状态”**（Frame类）分离开
     *      一个 SLAM 系统通常只有一个 Camera 对象（或者双目的一对），但会有成千上万个 Frame 对象。这种分离极大地节省了内存
     */ 
    Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);
    Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

    Vec2 camera2pixel(const Vec3 &p_c);
    Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

    // 这两个函数本质上就是把上面两步串起来：World -> Camera -> Pixel
    Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1);
    Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);
};

}  // namespace myslam
#endif  // MYSLAM_CAMERA_H
