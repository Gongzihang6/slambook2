/**
 * 实现了**视觉里程计（Visual Odometry）**的核心逻辑，包括特征提取、光流追踪、位姿估计、关键帧策略和三角化
 */

#include <opencv2/opencv.hpp>
#include <memory>
#include "myslam/algorithm.h"   // 通常包含通用的算法，如三角化函数 triangulation
#include "myslam/backend.h"     // 后端优化模块，前端会触发后端进行优化
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"   // 定义了图优化中使用的顶点（Vertex）和边（Edge）
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

/**
 * 这里使用了 GFTT (Good Features To Track) 特征点检测器。
 * 相比 ORB 或 SIFT，GFTT（Shi-Tomasi 角点）计算速度快，且角点稳定性好，梯度明显，非常适合 LK 光流法追踪
 * 
 * cv::GFTTDetector::create(maxCorners, qualityLevel, minDistance)
 * num_features: 最大角点数（如 150 个）
 * 0.01: 角点质量阈值
 * 20: 两个角点之间的最小像素距离（用于保证特征点分布均匀，不要挤在一起）
 */
Frontend::Frontend() {
    gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);

    // Config::Get 是一个单例模式的配置类，用于读取配置文件（如 .yaml），避免硬编码参数
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

/**
 * 这是前端处理每一帧图像的入口
 */
bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;     // 更新当前帧指针

    // switch-case 结构清晰地管理状态流转
    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();   // 如果处于初始化状态，执行双目初始化
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();        // 如果处于跟踪状态，执行跟踪逻辑
            break;
        case FrontendStatus::LOST:
            Reset();        // 如果丢失，尝试重置
            break;
    }

    last_frame_ = current_frame_;   // 处理完后，当前帧变成“上一帧”
    return true;
}

/**
 * 核心跟踪函数，负责计算当前帧的位姿
 */
bool Frontend::Track() {
    // 1、恒速模型预测位姿
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    // 2、光流法跟踪上一帧特征点
    int num_track_last = TrackLastFrame();
    // 3. 仅优化位姿 (Pose Only Optimization)已知3D地图点（来自上一帧）和当前帧的2D像素点，求解位姿。
    tracking_inliers_ = EstimateCurrentPose();

    // 4. 根据内点数量更新跟踪状态
    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    // 5、关键帧决策
    InsertKeyframe();
    // 6. 更新相对运动模型 (为下一帧预测做准备)
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}


/**
 * 决定是否将当前帧设为关键帧并更新地图
 * 关键帧选取的策略这里很简单——“跟丢了就插”。当跟踪的内点数少于阈值（num_features_needed_for_keyframe_），说明场景发生了较大变化，需要新的关键帧来记录新环境
 *      1、把当前帧转正为 KeyFrame
 *      2、补点: 因为旧的特征点不够了，需要 DetectFeatures 提取新角点
 *      3、三角化: 仅有左目的新角点是没深度的，需要 FindFeaturesInRight 找到右目匹配，然后 TriangulateNewPoints 恢复深度
 * 
 * 关键帧有什么用？关键帧在系统中承担了3个核心任务，普通帧是不做这些的
 *      1、补充新地图点（建图）
 *          只有关键帧会触发新特征的提取和三角化，普通帧只会用光流法去追踪已有的特征点，不会产生新的特征点和对应的地图点；
 *      2、参与后端优化（修正误差）
 *          后端只优化关键帧，后端优化的对象是Map中的active_keyframes_（活跃关键帧）和 active_landmarks_（活跃地图点）
 *      如果把每一帧都放进优化，计算量太大，实时性无法保证；通过只优化关键帧，我们维持了一个稀疏的位姿图，既保证了精度，又控制了计算规模。
 *      3、维持滑动窗口
 *          为了防止地图无限增长，Map类维护了一个滑动窗口（例如只保留最近的7个关键帧），在 ch13/src/map.cpp 的 RemoveOldKeyframe() 中：
 *      如果关键帧数量超过阈值，系统会根据距离（删除太近的或太远的）删除旧的关键帧，确保内存和计算负载恒定；
 */
bool Frontend::InsertKeyframe() {
    // num_features_needed_for_keyframe_ 是配置文件设定的阈值（例如 80）
    // tracking_inliers_ 是当前帧追踪到的上一帧特征点的数量
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // 如果还能追踪到很多点（比如 > 80 个），说明场景变化不大
        // 不需要插入关键帧，直接返回
        return false;
    }
    // 如果追踪到的点很少了（说明相机运动幅度大，或者转到了新场景）
    // 当前帧被“提拔”为关键帧
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);   // 将关键帧插入地图中

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();   // 建立 Feature 和 MapPoint 的关联
    DetectFeatures();  // detect new features，关键帧需要补充新的特征点

    // track in right image
    FindFeaturesInRight();      // 在右目寻找对应点（为了三角化）
    // triangulate map points   
    TriangulateNewPoints();     // 三角化产生新的地图点
    // update backend because we have a new keyframe
    backend_->UpdateMap();      // 通知后端进行优化

    if (viewer_) viewer_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

/**
 * 三角化新地图点
 */
int Frontend::TriangulateNewPoints() {
    // 双目相机的相对位姿
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    // T_wc (世界到相机) -> T_cw，获取当前帧到世界坐标系的变换
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;   
    // ... 遍历所有特征点 ...
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        // 只有那些还没有对应地图点，且右图有匹配的特征点才需要三角化
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化

            // 将像素坐标转为归一化相机坐标 (x, y, 1)
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};

            Vec3 pworld = Vec3::Zero();     // 初始化三角化结果

            // 调用三角化算法
            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();     // 初始化一个新地图点
                pworld = current_pose_Twc * pworld;     // pworld 是相机坐标系下的点，转到世界坐标系  
                new_map_point->SetPos(pworld);      // 设置新地图点位置
                new_map_point->AddObservation(      // 添加能够
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(
            std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) {
            // use project point
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit() {
    int num_features_left = DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}


/**
 * 提取新的特征点，在初始化和发现关键帧中使用
 * 当前图像中可能已经有一些特征点正在被追踪（从上一帧跟过来的）。我们不希望在这些已经有特征点的位置重复提取新的角点，因为那样会导致特征点重叠，浪费计算资源
 * 解决方案（掩膜策略）：
 *      1、初始化：先创建一张和原图一样大的全白图片（255 代表“允许检测区域”）
 *      2、涂黑：遍历所有已经存在并追踪良好的特征点，在这个 Mask 图片上，把这些点周围（比如 10x10 的方框）涂成黑色（0）（0 代表“禁止检测区域”）
 *      3、检测：调用 cv::GFTTDetector::detect(image, keypoints, mask)。OpenCV 会自动忽略 Mask 为 0 的区域，只在 Mask 为 255 的白色区域里找新角点
 */
int Frontend::DetectFeatures() {
    // 创建掩膜 (Mask)
    // 目的：为了不让新提取的特征点和已经追踪到的点重叠
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {     // 遍历当前帧左图中已有的特征点
        // 在已有特征点周围画一个矩形，涂黑 (0)，表示这块区域不许提点
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    // 使用带 Mask 的 GFTT 检测（只在白色255的地方提新点）
    gftt_->detect(current_frame_->left_img_, keypoints, mask);

    // 将新检测到的点加入 Frame
    int cnt_detected = 0;   // 记录检测到的特征点数量
    // 将左图中新检测到的特征点加入到features_left_
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

/**
 * 在左图中检测特征点后，使用光流法寻找特征点在右图中的位置
 * 从而实现三角化，确定新的地图点
 */
int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;

    // 遍历左图中的特征点
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);   // 取出特征点像素坐标
        auto mp = kp->map_point_.lock();
        if (mp) {
            // 如果有地图点（也就是之前的旧特征点，已经有地图点匹配关系了），投影到右目作为初值
            auto px = camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left iamge，如果没有地图点（也就是刚刚检测到的新特征点），使用和左图一样的像素坐标作为初值
            kps_right.push_back(kp->position_.pt);
        }
    }

    // 左图 -> 右图 光流追踪
    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,kps_right, 
        status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // 保存右目特征点
    int num_good_pts = 0;   // 计数器：记录成功匹配了多少对点
    // 遍历所有的点（status 的大小等于 kps_left 的大小）
    for (size_t i = 0; i < status.size(); ++i) {
        // 如果 status[i] 为 true (1)，说明第 i 个特征点在右图中找到了对应位置
        if (status[i]) {
            // 1. 创建 OpenCV 的 KeyPoint 对象
            // kps_right[i] 是光流算出来的亚像素坐标，7 是关键点直径（经验值）
            cv::KeyPoint kp(kps_right[i], 7);

            // 2. 创建自定义的 Feature 对象 (C++ 智能指针)
            // new Feature(current_frame_, kp) 会在堆上分配内存
            Feature::Ptr feat(new Feature(current_frame_, kp));

            // 3. 标记该特征点不属于左图 (是右图的)
            feat->is_on_left_image_ = false;
            // 4. 将该特征点存入当前帧的右目特征容器中
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            // 【关键 SLAM 设计】
            // 如果光流失败（status[i] == 0），我们在右目特征容器中推入一个 nullptr。
            // 为什么要存空指针？
            // 为了保持“索引对齐” (Index Alignment)。
            // 这样 features_left_[i] 和 features_right_[i] 永远是指向同一个 3D 点的观测。
            // 如果这里不 push nullptr，索引就乱了，后面三角化时就不知道谁对谁了。
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    // 1. 准备双目相机的位姿
    // camera_left_->pose() 通常是单位阵 (世界原点)，camera_right_->pose() 包含基线平移
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;      // 记录初始化地图点数量
    // 遍历左目每一个特征点
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        // --- 检查匹配关系 ---
        // 如果 features_right_[i] 是空指针，说明光流没追踪到这个点，无法三角化，跳过
        if (current_frame_->features_right_[i] == nullptr) continue;

        // 否则，就可以根据左目和右目特征点对应关系，三角化得地图点3D坐标
        // --- 坐标转换 (Pixel -> Camera) ---
        // 这一步将 2D 像素坐标 (u, v) 转换为 归一化相机坐标 (x, y, 1)
        // 数学公式：x = (u - cx) / fx,  y = (v - cy) / fy
        // 这里的 points 存储的是两个相机坐标系下的“射线方向向量”
        std::vector<Vec3> points{   // 三维点在左相机和右相机坐标系下的坐标
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        // --- 3. 执行三角化 & 深度检查 ---
        // 调用下面的 triangulation 函数。
        // pworld[2] > 0 是“手性约束” (Chirality Check)，确保点在相机前方，而不是后方。
        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            // --- 4. 创建地图点 (MapPoint) ---
            // 使用工厂模式创建一个新的 MapPoint 对象 (shared_ptr)
            auto new_map_point = MapPoint::CreateNewMappoint();
            // 设置地图点的 3D 坐标
            new_map_point->SetPos(pworld);

            // --- 5. 建立图结构 (Graph Connection) ---
            // A. 地图点记录谁（哪个特征点，包括左图和右图）观测到了它 (Observer Pattern)
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            // B. 特征点记录它对应哪个地图点
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;

            // 将新点加入全局地图
            map_->InsertMapPoint(new_map_point);
        }
    }
    // --- 6. 完成初始化 ---
    current_frame_->SetKeyFrame();          // 第一帧设为关键帧
    map_->InsertKeyFrame(current_frame_);   // 将关键帧插入地图
    backend_->UpdateMap();                  // 触发后端（虽然此时没什么可优化的，但建立了结构）

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam