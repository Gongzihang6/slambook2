//
// Created by gaoxiang on 19-5-4.
//
#include "myslam/visual_odometry.h"
#include <chrono>   // C++11 标准的时间库，用于计算每一帧的处理耗时
#include "myslam/config.h"

namespace myslam {

/**
 * 构造函数里没有任何逻辑，只保存了配置文件的路径
 * 设计哲学: 构造与初始化分离。构造函数应该很快，不要在里面做可能会失败的重操作（比如打开文件、申请大块内存）。
 * 重操作放在 Init() 里，方便通过返回值判断是否成功
 */
VisualOdometry::VisualOdometry(std::string &config_path)    // 使用冒号语法 : config_file_path_(config_path) 初始化成员变量
    : config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    // 1. 读取配置文件
    // 这是一个静态方法调用，单例模式的 Config 类被初始化
    if (Config::SetParameterFile(config_file_path_) == false) {     // 检查配置文件是否存在
        return false;
    }

    // 2. 初始化数据集
    // 从 Config 中读取 "dataset_dir" 字段，创建一个 Dataset 对象
    dataset_ = Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    // CHECK_EQ 是 glog 库的宏，相当于 assert。如果 dataset_->Init() 不返回 true，程序会直接奔溃报错
    CHECK_EQ(dataset_->Init(), true);

    // 3. 创建核心组件 (Create Components)
    // 这里并没有开始算法，只是申请了内存对象
    frontend_ = Frontend::Ptr(new Frontend);    // 创建前端
    backend_ = Backend::Ptr(new Backend);       // 创建后端
    map_ = Map::Ptr(new Map);                   // 创建地图
    viewer_ = Viewer::Ptr(new Viewer);          // 创建可视化器

    // 4. 建立连接 (Dependency Injection / Linking)
    // 这是最重要的一步！各模块原本是孤立的，现在通过 Set 函数把指针传给对方

    // 前端需要后端（发现关键帧要触发优化）、地图（要存点）、可视化器（要更新画面）、相机（要投影）
    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    // 后端需要地图（要优化里面的点）、相机（计算重投影误差需要内参）
    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    // 可视化器需要地图（画出点云和轨迹）
    viewer_->SetMap(map_);

    return true;    // 组装完成，返回成功
}

void VisualOdometry::Run() {
    // 1. 死循环，直到 Step() 返回 false
    while (1) {
        LOG(INFO) << "VO is running";
        if (Step() == false) {
            break;      // 数据集读完了，或者出错，跳出循环
        }
    }
    // 2. 退出前的清理工作
    backend_->Stop();   // 通知后端线程停止循环，并 join 等待其结束
    viewer_->Close();   // 关闭可视化窗口

    LOG(INFO) << "VO exit";
}

// 单步执行函数 Step() —— 核心工作流
bool VisualOdometry::Step() {
    // 1. 从数据集中泵出新的一帧
    Frame::Ptr new_frame = dataset_->NextFrame();
    // 如果返回空指针，说明数据读完了，返回 false 告知 Run() 退出
    if (new_frame == nullptr) return false;

    // 2. 计时开始
    auto t1 = std::chrono::steady_clock::now();
    // 3. 【核心调用】把帧喂给前端
    // 整个 SLAM 的数学运算都在这一行里被触发
    // Frontend 会追踪、定位、建图、并在需要时通知 Backend
    bool success = frontend_->AddFrame(new_frame);
    // 4. 计时结束
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    return success;
}

}  // namespace myslam
