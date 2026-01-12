//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 * 
 * 线性三角化 (SVD 解法)
 * 输入：poses (两个相机的位姿 T), points (归一化平面坐标 x,y,1)
 * 输出：pt_world (三角化后的 3D 坐标)
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {
    // --- 1. 构建方程组 Ax = 0 ---
    // A 的大小是 (2 * 相机数) 行，4 列。这里是双目，所以是 4x4 矩阵。
    // 待求解的 x 是 3D 点的齐次坐标 (X, Y, Z, 1)。
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();    // DLT 方法其实不需要 b，因为是齐次方程 Ax=0

    for (size_t i = 0; i < poses.size(); ++i) {
        // 获取 3x4 的投影矩阵 P = [R|t]
        Mat34 m = poses[i].matrix3x4();
        // 填充 A 矩阵
        // 第一行：u * P_row3 - P_row1 = 0
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        // 第二行：v * P_row3 - P_row2 = 0
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }

    // --- 2. SVD 分解求解 ---
    // 求解 Ax=0 的最小二乘解，等价于对 A 进行 SVD 分解后，
    // 最小奇异值对应的右奇异向量 (V 矩阵的最后一列)
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    // svd.matrixV().col(3) 取出最后一列（因为是 4x4，索引 0,1,2,3）
    // 这一列向量是 (X, Y, Z, W)
    // 我们需要非齐次坐标，所以除以 W (也就是 svd.matrixV()(3, 3))
    // .head<3>() 取出前三个元素 (X/W, Y/W, Z/W)
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // 解质量好，成功
        return true;
    }
    // 如果比率很大，说明最小奇异值和倒数第二小没拉开差距，解在“晃动”，不可靠。
    return false;
}

// converters
inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
