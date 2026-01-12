#pragma once
#ifndef MAP_H   // 意思：预处理器，请检查一下，MAP_H 这个记号（宏）之前有没有被定义过？
#define MAP_H   // 意思：如果没有被定义过（说明是第一次引用这个文件），那我现在就定义 MAP_H 这个记号，然后继续处理下面的代码
/**
 * 作用：防止头文件被重复包含（重复定义）。
 * 问题背景： 在你的项目中，关系错综复杂：
 *      frontend.h 可能 #include "map.h"
 *      backend.h 可能也 #include "map.h"
 *      然后 visual_odometry.cpp 同时引用了 #include "frontend.h" 和 #include "backend.h"。
 * 如果没有保护符，visual_odometry.cpp 在编译预处理阶段，会先把 frontend.h 里的 Map 类定义拷贝进来，
 * 然后又把 backend.h 里的 Map 类定义拷贝进来。 结果编译器会看到两份一模一样的 class Map { ... } 定义，
 * 它会直接报错：“Refinition of class Map”（类 Map 重复定义）。
 * 
 * 执行流程：
 * 编译器第一次遇到 map.h：检查 MAP_H？没定义。-> 好，定义 MAP_H，并拷贝 class Map 的代码。
 * 编译器第二次遇到 map.h（通过另一个文件间接引用）：检查 MAP_H？发现已经定义过了。-> 直接跳到 #endif，完全忽略中间的所有代码。
 */



#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

// 防止命名冲突（污染）
namespace myslam {

/**
 * @brief 地图
 * Map 类依赖于 Frame（关键帧）和 MapPoint（路标点），这三者构成了 SLAM 的核心数据模型：地图由帧和点组成。
 * 和地图的交互：前端调用InsertKeyframe和InsertMapPoint插入新帧和地图点，后端维护地图的结构，判定outlier/剔除等等
 */
class Map {
public:
    // 1. Eigen 内存对齐宏
    /**
     * 类成员（虽然这里主要在 Frame 和 MapPoint 中，但 Map 可能作为管理类被传递）可能涉及 Eigen 的固定大小向量运算。
     * 在 64 位系统下，Eigen 需要 16 字节内存对齐来执行 SIMD（单指令多数据）加速。如果不对齐，程序可能会崩溃（Segfault）
     */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // 2. 智能指针别名
    // 整个地图由智能指针管理，这意味着只要地图存在，里面的帧和点就不会被销毁；一旦从地图中移除且没有其他地方引用，内存自动释放
    typedef std::shared_ptr<Map> Ptr;
    // 3. 核心存储容器：哈希表
    // Key: unsigned long (ID), Value: 智能指针
    /**
     * 为什么不用std::vector ？
     * 因为 SLAM 运行过程中会不断剔除无效点或旧帧，ID 并不是连续的（比如保留了 ID 1, 5, 100）
     * 哈希表允许我们通过 ID 以 O(1) 的平均复杂度快速找到对应的对象，而不需要遍历整个列表
     */
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

    Map() {}

    /**
     * 这两个函数通常由前端调用
     * 当某一帧被判定为关键帧时，前端调用InsertKeyFrame，将关键帧加入地图中
     * 当三角化产生新点时，前端调用InsertMapPoint，将新点加入地图中
     * Map是被动的，它不知道什么时候该加数据，它只负责存储
     */
    // 增加一个关键帧
    void InsertKeyFrame(Frame::Ptr frame);
    // 增加一个地图顶点
    void InsertMapPoint(MapPoint::Ptr map_point);


    /**
     * SLAM是典型的多线程系统
     *      线程 A（后端）：正在优化，可能会删除某个点，或者修改点的坐标
     *      线程 B（可视化）：正在遍历所有点进行绘图
     * 如果不加锁，线程 B 读到一半数据被线程 A 删了，程序就会崩溃。
     * 这里加锁是为了保证获取到的 landmarks_ 是一个完整的、未被修改的状态
     */ 
    // 获取所有地图点
    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    // 获取所有关键帧
    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }


    /**
     * SLAM 设计哲学：滑动窗口 (Sliding Window)，这就是所谓的局部地图
     *      landmarks_ 存的是全量历史数据（用于回环检测或最终地图保存）
     *      active_landmarks_ 存的是当前附近的数据（用于后端实时优化）
     * 
     * 为什么要分两个？ 随着时间推移，全量地图会越来越大（几万个点）。
     * 如果后端每次都优化所有点，计算量是 $O(N^3)$，系统会越来越慢卡死。通过只优化“激活”的局部点，保证了 SLAM 的实时性
     */
    // 获取激活地图点
    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    // 获取激活关键帧
    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    /**
     * 清理map中观测数量为零的点
     * 
     * 在 SLAM 中，有些点可能是误匹配产生的（Outliers），或者只被看到一次就再也没出现过。
     * 这些点对定位没有帮助，反而占用内存和计算资源。这个函数负责把它们从地图中剔除
     */
    void CleanMap();

private:
    // 将旧的关键帧置为不活跃状态
    /**
     * 当新的关键帧进来，如果“激活关键帧”的数量超过了设定值（比如 7 个），就需要把最老的或者距离当前最远的关键帧移出“激活组”。
     * 这保证了后端优化的规模始终维持在一个常数级别（比如始终只优化最近的 7 帧），从而实现长时间稳定运行。
     */
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    LandmarksType landmarks_;         // all landmarks（全集，所有地图点）
    LandmarksType active_landmarks_;  // active landmarks（子集，当前优化的局部地图点）
    KeyframesType keyframes_;         // all key-frames（全集，所有关键帧）
    KeyframesType active_keyframes_;  // all key-frames（子集，当前优化的关键帧）

    Frame::Ptr current_frame_ = nullptr;

    // settings
    // 这是一个经验值。窗口太小，约束不够，容易漂移；窗口太大，计算太慢。7 是一个在轻量级 VO 中常见的平衡点
    int num_active_keyframes_ = 7;  // 激活的关键帧数量(窗口大小)
};
}  // namespace myslam

#endif  // MAP_H
