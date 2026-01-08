#ifndef ROTATION_H
#define ROTATION_H

#include <algorithm>
#include <cmath>
#include <limits>

//////////////////////////////////////////////////////////////////
// math functions needed for rotation conversion. 旋转变换需要的数学方程

// dot and cross production     点积和叉积

template<typename T>
inline T DotProduct(const T x[3], const T y[3]) {   // 点积，两个向量对应元素乘积之和
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template<typename T>
inline void CrossProduct(const T x[3], const T y[3], T result[3]) { // 叉积，可以用反对称矩阵乘法推导
    result[0] = x[1] * y[2] - x[2] * y[1];
    result[1] = x[2] * y[0] - x[0] * y[2];
    result[2] = x[0] * y[1] - x[1] * y[0];
}


//////////////////////////////////////////////////////////////////


// Converts from a angle anxis to quaternion : 
template<typename T>
inline void AngleAxisToQuaternion(const T *angle_axis, T *quaternion) {
    const T &a0 = angle_axis[0];
    const T &a1 = angle_axis[1];
    const T &a2 = angle_axis[2];
    const T theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    if (theta_squared > T(std::numeric_limits<double>::epsilon())) {
        const T theta = sqrt(theta_squared);
        const T half_theta = theta * T(0.5);
        const T k = sin(half_theta) / theta;
        quaternion[0] = cos(half_theta);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    } else { // in case if theta_squared is zero
        const T k(0.5);
        quaternion[0] = T(1.0);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

template<typename T>
inline void QuaternionToAngleAxis(const T *quaternion, T *angle_axis) {
    const T &q1 = quaternion[1];
    const T &q2 = quaternion[2];
    const T &q3 = quaternion[3];
    const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

    // For quaternions representing non-zero rotation, the conversion
    // is numercially stable
    if (sin_squared_theta > T(std::numeric_limits<double>::epsilon())) {
        const T sin_theta = sqrt(sin_squared_theta);
        const T &cos_theta = quaternion[0];

        // If cos_theta is negative, theta is greater than pi/2, which
        // means that angle for the angle_axis vector which is 2 * theta
        // would be greater than pi...

        const T two_theta = T(2.0) * ((cos_theta < 0.0)
                                      ? atan2(-sin_theta, -cos_theta)
                                      : atan2(sin_theta, cos_theta));
        const T k = two_theta / sin_theta;

        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else {
        // For zero rotation, sqrt() will produce NaN in derivative since
        // the argument is zero. By approximating with a Taylor series,
        // and truncating at one term, the value and first derivatives will be
        // computed correctly when Jets are used..
        const T k(2.0);
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }

}

/**
 * AngleAxis 就是旋转向量，它的单位方向向量描述了旋转轴；该向量的模长描述了旋转的角度；
 * 这个函数的作用是将旋转向量作用于输入点pt，将旋转之后的点坐标写入到result中
 * 
 * 大角度：使用 $SO(3)$ 上的精确解析解（罗德里格斯公式）。小角度：使用 $\mathfrak{so}(3)$ 上的切空间近似（一阶泰勒展开）。
 * 为什么要分情况讨论？罗德里格斯公式中需要计算归一化的轴 $\mathbf{w} = \frac{\mathbf{v}}{\theta}$。如果 $\theta \approx 0$，就会出现 除以零 的错误。
 *                   当旋转角 $\theta$ 极小时，表示几乎没有旋转，或者是微小的扰动，就更适合使用李代数近似
 */
template<typename T>
inline void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3]) {
    // 1. 计算旋转向量模长的平方：theta2 = theta^2
    const T theta2 = DotProduct(angle_axis, angle_axis);

    // 2. 判断是否远离零点 (machine epsilon 是浮点数的最小分辨率)
    if (theta2 > T(std::numeric_limits<double>::epsilon())) {
        // Away from zero, use the rodriguez formula
        //
        //   result = pt cos(theta) +
        //            (w x pt) * sin(theta) +
        //            w (w . pt) (1 - cos(theta))
        //
        // We want to be careful to only evaluate the square root if the
        // norm of the angle_axis vector is greater than zero. Otherwise
        // we get a division by zero.
        // 其中w是旋转轴（即T除以模长后的单位方向向量）


        // 3. 计算角度 theta 和三角函数
        const T theta = sqrt(theta2);
        const T costheta = cos(theta);
        const T sintheta = sin(theta);

        // 4. 计算 1/theta，用于归一化轴
        const T theta_inverse = 1.0 / theta;

        // 5. 归一化，得到旋转轴 w (单位向量)
        const T w[3] = {angle_axis[0] * theta_inverse,
                        angle_axis[1] * theta_inverse,
                        angle_axis[2] * theta_inverse};

        // Explicitly inlined evaluation of the cross product for
        // performance reasons.
        /*const T w_cross_pt[3] = { w[1] * pt[2] - w[2] * pt[1],
                                  w[2] * pt[0] - w[0] * pt[2],
                                  w[0] * pt[1] - w[1] * pt[0] };*/
        T w_cross_pt[3];    // 存储 w × p 的叉积结果
        CrossProduct(w, pt, w_cross_pt);

        // 7. 计算 w(w . p)(1 - cos) 的标量部分
        // DotProduct(w, pt) 对应公式里的 (w . p)
        const T tmp = DotProduct(w, pt) * (T(1.0) - costheta);
        //    (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);

        // 8. 组合最终结果 (对应罗德里格斯公式的三项)
        // result = p * cos + (w x p) * sin + w * tmp
        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    } else {
        // Near zero, the first order Taylor approximation of the rotation
        // matrix R corresponding to a vector w and angle w is
        //
        //   R = I + hat(w) * sin(theta)
        //
        // But sintheta ~ theta and theta * w = angle_axis, which gives us
        //
        //  R = I + hat(w)
        //
        // and actually performing multiplication with the point pt, gives us
        // R * pt = pt + w x pt.
        //
        // Switching to the Taylor expansion near zero provides meaningful
        // derivatives when evaluated using Jets.
        //
        // Explicitly inlined evaluation of the cross product for
        // performance reasons.
        /*const T w_cross_pt[3] = { angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                                  angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                                  angle_axis[0] * pt[1] - angle_axis[1] * pt[0] };*/
        
        // 9. 直接计算 angle_axis x pt
        // 注意：这里不需要归一化，直接用原始向量做叉积
        T w_cross_pt[3];
        CrossProduct(angle_axis, pt, w_cross_pt);

        // 10. 结果 = 原点 + 叉积修正量
        // result = p + v x p
        result[0] = pt[0] + w_cross_pt[0];
        result[1] = pt[1] + w_cross_pt[1];
        result[2] = pt[2] + w_cross_pt[2];
    }
}

#endif // rotation.h