#pragma once
#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "myslam/backend.h"
#include "myslam/common_include.h"
#include "myslam/dataset.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"

namespace myslam {

/**
 * VO 对外接口
 * 作为一个“大管家”，负责组装各个模块（Frontend, Backend, Map, Viewer），并调度数据的流转
 * 这是用户（比如 run_kitti_stereo.cpp）唯一需要直接交互的类。用户不需要知道什么是光流、什么是 G2O，只需要创建这个对象并调用 Run() 即可
 */
class VisualOdometry {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualOdometry> Ptr;

    // constructor with config file，构造函数只接收配置文件路径，通常只做简单的变量赋值，不做繁重的初始化工作（如加载大模型、打开数据集）
    VisualOdometry(std::string &config_path);

    /**
     * do initialization things before run
     * @return true if success
     * SLAM 设计哲学: 将 构造（Construction） 和 初始化（Initialization） 分离
     * 在 Init 中，系统会做以下几件大事（参考 src/visual_odometry.cpp）
     *      加载配置：调用 Config::SetParameterFile
     *      实例化模块：new Frontend, new Backend, new Map, new Viewer
     *      建立连接（Dependency Injection）：这是最关键的一步
     *          把 Map 指针塞给 Frontend 和 Backend
     *          把 Backend 指针塞给 Frontend
     *          把 Viewer 指针塞给 Frontend
     * 这样各个模块才能互相通信
     */
    bool Init();

    /**
     * start vo in the dataset
     * 这是一个死循环（Loop），它会不断调用 Step() 直到数据集读完或用户强制退出
     * 设计目的: 为用户提供“一键启动”的傻瓜式接口
     */
    void Run();

    /**
     * Make a step forward in dataset
     * 设计目的: 提供“单步执行”的能力
     * 很多时候我们需要调试，或者希望将 SLAM 嵌入到其他系统（如 ROS）中，此时 ROS 的回调函数会驱动每一帧的处理。
     * Step() 就是为了这种场景设计的：处理一帧数据，然后返回
     */
    bool Step();

    // 获取前端状态
    // 作用: 向外界暴露系统当前的健康状况（初始化中？跟踪正常？跟丢了？）。外部程序可以根据这个状态决定是否重置系统或发出警告
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }


/**
 * VisualOdometry 拥有所有核心模块的 shared_ptr。这意味着 VisualOdometry 对象的生命周期决定了整个 SLAM 系统的生命周期
 * 当 VisualOdometry 被析构时，引用计数减少，Map、Frontend、Backend 等都会随之自动释放（前提是线程已经 join 并且没有循环引用）
 */
private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    // dataset
    Dataset::Ptr dataset_ = nullptr;
};
}  // namespace myslam

#endif  // MYSLAM_VISUAL_ODOMETRY_H
