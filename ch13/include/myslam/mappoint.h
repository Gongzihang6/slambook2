// 防止头文件被重复包含的标准做法
#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

// 包含常用的头文件（如 Eigen, OpenCV, std::vector 等的定义）
#include "myslam/common_include.h"

namespace myslam {

// 前向声明（Forward Declaration）
// 告诉编译器 Frame 和 Feature 这两个类存在，但具体的定义在别处。
// 这样做是为了解决“循环引用”的问题（MapPoint 引用 Feature，Feature 也引用 MapPoint）。
struct Frame;

struct Feature;

/**
 * 路标点类
 * 特征点在三角化之后形成路标点
 */
struct MapPoint {
public:
    // 1. Eigen 内存对齐宏
    // 原因：MapPoint 内部包含 Eigen::Vector3d (Vec3) 类型的成员变量。
    // 在 64 位系统中，Eigen 为了利用 SSE/AVX 指令集加速运算，要求定长向量（如 Vector3d）在内存中必须是 16 字节对齐的。
    // 如果直接使用 new MapPoint()，C++ 默认的 new 操作符不保证这种对齐，可能导致程序崩溃（Segfault）。
    // 这个宏重载了 new/delete 操作符，确保分配的内存是对齐的。
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;        // Eigen内存对齐宏

    // 2. 智能指针定义
    // 使用 shared_ptr 管理 MapPoint 的生命周期，防止内存泄漏。
    // 当所有持有该点的 Frame 或 Map 释放它时，它会自动销毁。
    typedef std::shared_ptr<MapPoint> Ptr;

    // 作用：全局唯一标识符。
    // 在后端优化（G2O）中，我们需要根据 ID 来寻找对应的顶点（Vertex）。
    unsigned long id_ = 0;  // ID

    // 作用：标记是否为外点（异常点）。
    // 在后端优化（Bundle Adjustment）或前端跟踪时，如果重投影误差过大（卡方检验失败），
    // 算法会将该点标记为 outlier，下次就不再使用它进行位姿估计，避免带偏结果
    bool is_outlier_ = false;

    // 作用：路标点在世界坐标系下的 3D 坐标 $(x, y, z)$。
    // 这是 SLAM 建图产生的直接结果。
    Vec3 pos_ = Vec3::Zero();  // Position in world

    // 作用：数据互斥锁。
    // SLAM 通常是多线程系统（前端线程追踪，后端线程优化，可视化线程绘图）。
    // 当后端线程正在修改 pos_（优化位置）时，前端线程可能正在读取 pos_（用于投影匹配）。
    // 为了防止数据竞争（Data Race），读写数据时必须加锁。
    std::mutex data_mutex_;

    // 作用：被观测次数。
    // 这个点被多少个特征点（Feature）匹配到了。
    // 次数越多，说明这个点越稳定、越可靠，是一个“好点”。
    // 地图清理机制会利用这个值删掉那些偶尔被看到一次就不见了的“不稳定点”。
    int observed_times_ = 0;  // being observed by feature matching algo.

    // 为什么用 weak_ptr（弱指针）？
    // 这是一个经典的“循环引用”问题：
    // MapPoint 持有 Feature 的列表 (observations_)
    // Feature 持有 MapPoint 的指针 (map_point_)
    // 如果都用 shared_ptr，双方互相抓着对方不放，引用计数永远不为 0，导致内存泄漏。
    // weak_ptr 不增加引用计数，只起“观察”作用，解决了这个问题。
    std::list<std::weak_ptr<Feature>> observations_;

    // 构造函数
    MapPoint() {}
    // 具体的实现通常在 .cpp 文件中。
    // 初始化时赋予它一个 ID 和初始的 3D 坐标（通常来自三角化结果）。
    MapPoint(long id, Vec3 position);

    // 线程安全的读写函数
    Vec3 Pos() {
        // 使用 unique_lock 自动加锁。
        // 当函数结束（作用域结束）时，lck 析构，自动解锁。
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    /**
     * 后端优化线程计算出了更准确的坐标，调用 SetPos 更新；
     * 此时可视化线程调用 Pos 想要画图，锁机制保证可视化线程要么拿到旧值，要么拿到新值，绝不会拿到一个“写了一半”的坏数据
     */
    void SetPos(const Vec3 &pos) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    };

    // 观测关系管理

    /**
     * 当一个新的关键帧进来，并通过特征匹配发现某些特征点对应这个MaoPoint时，就需要调用AddObservation
     * 这建立起了3D点与2D特征的联系（Graph Optimization中的边Edge）
     */
    void AddObservation(std::shared_ptr<Feature> feature) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }

    // 如果某个特征点被判定为误匹配，或者对应的帧被删除了，需要移除观测关系
    void RemoveObservation(std::shared_ptr<Feature> feat);

    std::list<std::weak_ptr<Feature>> GetObs() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    // factory function，这是一个静态函数，用于创建新的 MapPoint 对象
    /**
     * 为什么不直接 new？
     * 为了统一管理 ID。这个函数内部通常维护一个静态计数器，每创建一个点，ID 自动加 1。这样使用者不需要自己去操心 ID 是多少，也不会出现 ID 重复的情况。
     * 强制返回 shared_ptr，确保外部使用时符合内存管理规范
     */
    static MapPoint::Ptr CreateNewMappoint();
};
}  // namespace myslam

#endif  // MYSLAM_MAPPOINT_H
