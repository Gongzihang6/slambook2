/**
 * Frontend 类是 SLAM 系统的**“排头兵”**：
 * 时效性 (Speed): 
 *      后端优化可能很慢（几百毫秒），但前端必须跟上相机帧率（30FPS，即 33ms 一帧）。所以前端只做轻量级的计算（光流、仅位姿优化）。
 * 数据过滤 (Data Filtering): 
 *      视频流有大量冗余信息。前端通过关键帧策略，把 90% 的普通帧过滤掉，只把最有价值的 10% 关键帧喂给后端和地图，保证系统不会因为内存耗尽而崩溃。
 * 初值提供 (Initialization): 
 *      后端是非线性优化，非常依赖初值。前端负责提供准确的初值（位姿和路标点），如果没有前端，后端根本无法启动计算。
 */

#pragma once
#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include <opencv2/features2d.hpp>   // 前端需要处理图像特征（如提取角点），所以需要 OpenCV 的特征模块

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

/**
 * Frontend 类中只需要存它们的指针（shared_ptr）。
 * 使用前向声明可以避免循环引用（例如 Frontend 引用 Backend，Backend 又引用 Frontend），并减少编译时间。
 * 只有在 .cpp 文件中真正调用它们的方法时，才需要 include 头文件
 */
class Backend;
class Viewer;

/**
 * SLAM中前端通常被设计为一个有限状态机
 *      INITING: 系统刚启动，还没有地图，正在尝试初始化（比如双目正在找第一对匹配，或者单目正在进行纯旋转检测）
 *      TRACKING_GOOD: 追踪正常，匹配到的特征点数量充足，位姿估计可信
 *      TRACKING_BAD: 还能勉强追踪，但特征点很少，可能快要跟丢了（风险预警）
 *      LOST: 彻底跟丢了（比如相机被遮挡，或移动太快出现模糊）。此时通常需要触发 重定位（Relocalization） 模块
 */
enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

/**
 * 前端
 * 估计当前帧Pose，在满足关键帧条件时向地图加入关键帧并触发优化
 */
class Frontend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    // 外部接口，添加一个帧并计算其定位结果
    /**
     * 这是前端对外的唯一主要入口
     * 对于 Dataset 类或 VisualOdometry 主线程来说，它们不需要知道前端内部是怎么用光流还是特征匹配的
     * 它们只负责不断地把新的图像（封装成 Frame）塞给 AddFrame
     * 
     * 返回值: true 表示处理成功，false 表示失败（可能初始化失败或跟丢了）
     */
    bool AddFrame(Frame::Ptr frame);

    /**
     * 设计模式，前端本身不负责创建地图、后端或可视化器，而是通过外部（通常在 VisualOdometry::Init 中）创建好后“塞”进来的。
     * 好处: 模块解耦。比如测试时，可以给前端塞一个空的 MockBackend，只测试追踪逻辑。
     */
    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }

    FrontendStatus GetStatus() const { return status_; }

    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        camera_left_ = left;
        camera_right_ = right;
    }

private:
    /**
     * Track in normal mode
     * @return true if success
     * 正常追踪模式的主函数。当状态不是初始化时，AddFrame 会调用它。
     * 它通常包含：位姿预测 -> 追踪上一帧 -> 估计位姿 -> 关键帧判断 这一连串动作
     */
    bool Track();

    /**
     * Reset when lost
     * @return true if success
     */
    bool Reset();

    /**
     * Track with last frame
     * @return num of tracked points
     * 方法: 通常使用 LK 光流法 (Lucas-Kanade Optical Flow)
     * 逻辑: 已知上一帧的特征点位置，推测它们在当前帧的位置。这是 2D-2D 的像素级追踪，速度极快
     */
    int TrackLastFrame();

    /**
     * estimate current frame's pose
     * @return num of inliers
     * PnP (Perspective-n-Point) 问题
     * 输入: 上一帧（或局部地图）已知的 3D 地图点 + TrackLastFrame 找到的当前帧 2D 像素点
     * 工具: 通常使用 G2O 进行图优化（Pose Only Optimization），只优化当前帧的 $T_{cw}$，固定地图点 $P_w$ 不变
     */
    int EstimateCurrentPose();

    /**
     * set current frame as a keyframe and insert it into backend
     * @return true if success
     * 关键帧选择策略
     * 如果当前帧跟上一帧很像（追踪到的点很多，运动很小），就扔掉（只做定位，不建图）
     * 如果运动很大或特征点跟丢了很多，就把当前帧设为关键帧，加入后端优化，扩展地图
     */
    bool InsertKeyframe();

    /**
     * Try init the frontend with stereo images saved in current_frame_
     * @return true if success
     * 系统启动的第一步，SLAM 系统启动时是没有地图的。
     * 这个函数利用双目相机的左右图像，通过视差直接算出第一批 3D 点，建立初始地图
     */
    bool StereoInit();

    /**
     * Detect features in left image in current_frame_
     * keypoints will be saved in current_frame_
     * @return
     * 提取新特征（当特征点不够时调用，通常用 GFTT）
     */
    int DetectFeatures();   

    /**
     * Find the corresponding features in right image of current_frame_
     * @return num of features found
     * 在右目找对应点（为了三角化深度）
     */
    int FindFeaturesInRight();

    /**
     * Triangulate the 2D points in current frame
     * @return num of triangulated points
     * 三角化恢复深度，生成新的 MapPoint
     */
    int TriangulateNewPoints();     // 检测特征点+在右目找对应点+三角化恢复深度，这3个函数是建图的核心步骤，通常只在关键帧上执行

    /**
     * Build the initial map with single image
     * @return true if succeed
     */
    bool BuildInitMap();

    /**
     * Set the features in keyframe as new observation of the map points
     * 作用: 维护图结构。建立 Feature（2D）和 MapPoint（3D）之间的指针关联，这构成了 Pose Graph 中的“边”
     */
    void SetObservationsForKeyFrame();

    // data
    FrontendStatus status_ = FrontendStatus::INITING;

    Frame::Ptr current_frame_ = nullptr;  // 当前帧
    Frame::Ptr last_frame_ = nullptr;     // 上一帧（用于光流追踪的参考）
    Camera::Ptr camera_left_ = nullptr;   // 左侧相机
    Camera::Ptr camera_right_ = nullptr;  // 右侧相机

    Map::Ptr map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;

    // SLAM 技巧：恒速模型 (Constant Velocity Model)。
    // 我们假设相机在极短时间内是匀速运动的。因此，可以用上一帧的相对运动 $T_{k, k-1}$ 来预测当前帧的位姿
    SE3 relative_motion_;  // 当前帧与上一帧的相对运动，用于估计当前帧pose初值

    // 追踪到的有效点（内点）数量。这是判断系统健康状况（Status）和是否需要插入关键帧的核心指标
    int tracking_inliers_ = 0;  // inliers, used for testing new keyframes

    // params
    int num_features_ = 200;                        // 每一帧期望保持的特征点数量
    int num_features_init_ = 100;                   // 初始化需要的最小点数
    int num_features_tracking_ = 50;                // 追踪良好的阈值
    int num_features_tracking_bad_ = 20;            // 追踪丢失的阈值
    int num_features_needed_for_keyframe_ = 80;     // 关键帧判定阈值（少于80个就插关键帧）

    // utilities
    // GFTT (Good Features To Track)，也就是 Shi-Tomasi 角点
    // 选择理由: 相比 ORB，GFTT 计算更快，且角点分布更均匀，非常适合光流法（Optical Flow）追踪。因为这个项目是基于光流的前端，而不是基于描述子匹配的前端。
    cv::Ptr<cv::GFTTDetector> gftt_;  // feature detector in opencv
};

}  // namespace myslam

#endif  // MYSLAM_FRONTEND_H
