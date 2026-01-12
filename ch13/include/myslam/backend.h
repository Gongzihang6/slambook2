/**
 * 前端（Frontend）负责快速地跟踪特征点并给出粗略的轨迹，但误差会积累（漂移）。
 * 后端则在一个独立的线程中，利用所有观测数据进行非线性优化（Bundle Adjustment, BA），把之前的轨迹和地图修得更准
 */

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
class Map;      // 前向声明，虽然上面include了map.h，但在某些循环引用的复杂依赖关系中，前向声明能减少编译依赖耦合

/**
 * 后端
 * 有单独优化线程，在Map更新时启动优化
 * Map更新由前端触发
 */ 
class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    // 构造函数中启动优化线程并挂起
    // 构造函数不仅是初始化变量，这里还干了一件大事：启动了一个新的线程（std::thread）。这个线程创建后通常会立刻进入 wait（挂起）状态，等待前端派活
    Backend();

    // 设置左右目的相机，用于获得内外参
    // 后端优化计算误差需要使用相机的内外参数
    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        cam_left_ = left;
        cam_right_ = right;
    }

    // 设置地图，后端需要拿到“地图”这个共享资源的指针，以便从中提取活跃的关键帧和地图点进行优化
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    // 触发地图更新，启动优化，当前端（Frontend）插入了一个新的关键帧（InsertKeyFrame）后，会调用这个函数
    // 这个函数内部会调用 map_update_.notify_one()。这就好比前端按了一下门铃，告诉在后台睡觉的后端线程：“醒醒，来新活了（新关键帧来了），快起来优化一下。”
    void UpdateMap();

    // 关闭后端线程
    // 系统退出时调用，将线程标志位设为停止，并唤醒线程让其自然退出循环，最后 join() 等待线程销毁，避免资源泄露
    void Stop();

private:
    // 后端线程
    /**
     * 后端线程的主体函数，通常是一个while(running)循环
     * 1、循环检查是否有新任务
     * 2、如果没有任务，调用 condition_variable.wait() 挂起（不占用 CPU）
     * 3、一旦被 UpdateMap() 唤醒，就调用 Optimize() 执行优化
     */
    void BackendLoop();

    // 对给定关键帧和路标点进行优化，输出优化后的位姿和点坐标，直接更新回 Map 中。这就修正了前端的漂移
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    // 线程同步成员变量
    std::shared_ptr<Map> map_;
    std::thread backend_thread_;    // 实际的后端线程句柄
    std::mutex data_mutex_;         // 互斥锁，保护共享数据，配合条件变量使用

    /**
     * 条件变量，这是实现异步处理的关键
     * SLAM 前端要求极高的实时性（每秒 30 帧），不能被后端（可能耗时 0.1 秒甚至更久）阻塞
     * 因此前端只负责“通知”（Notify），后端负责“等待并处理”（Wait & Process）。条件变量是实现这种 事件驱动 机制的最佳工具
     */
    std::condition_variable map_update_;

    /**
     * C++原子变量，线程安全的标志位
     * 用于控制 BackendLoop 的 while 循环何时结束。使用 atomic 避免了读写 bool 变量时的数据竞争，比加锁更高效
     */
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
};

}  // namespace myslam

#endif  // MYSLAM_BACKEND_H