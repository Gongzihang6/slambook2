/**
 * @file backend.cpp
 * 这个文件实现了SLAM后端优化的核心逻辑，主要任务是利用G2O库构建并求解一个BA（Bundle Adjustment）优化问题
 * 后端优化的目标是同时优化相机位姿和地图点位置，以减少重投影误差，提高整体SLAM系统的精度和鲁棒性
 */

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {

Backend::Backend() {
    // 1. 原子变量初始化
    // backend_running_ 是一个 std::atomic<bool>，用于控制线程循环的启停。
    // 使用 .store(true) 是线程安全的赋值操作
    backend_running_.store(true);

    // 2. 启动后端线程
    // std::thread 创建一个新线程。
    // std::bind(&Backend::BackendLoop, this) 将成员函数 BackendLoop 和当前对象指针 this 绑定，
    // 生成一个可调用对象（Functor），作为线程的入口函数。
    // 线程一旦创建，就会立即开始执行 BackendLoop() 中的代码。
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

// --- 生产者 (Frontend 调用) ---
void Backend::UpdateMap() {
    // 加锁：确保同一时间只有一个线程能修改条件变量相关的状态
    std::unique_lock<std::mutex> lock(data_mutex_);
    // 发送信号：唤醒正在 wait 的后端线程
    // 告诉它：“嘿，来新活了（有新关键帧加入了），快起来干活！”
    map_update_.notify_one();
}

// --- 系统控制 (主线程调用) ---
void Backend::Stop() {
    // 1. 设置标志位为 false，让 BackendLoop 跳出 while 循环
    backend_running_.store(false);
    // 2. 发送信号：万一线程正在睡觉（wait），必须把它叫醒，它才能检查到 running 变 false 了
    map_update_.notify_one();
    // 3. 等待线程结束：阻塞主线程，直到 backend_thread_ 执行完毕
    // 这是一个好习惯，防止主程序退出了但子线程还在跑（导致段错误）。
    backend_thread_.join();
}

/**
 * BackendLoop - 消费者线程主循环
 * 后端只优化“激活”的关键帧和地图点，采用滑动窗口优化策略，保证实时性
 * 1. 等待信号 (Wait)：调用条件变量的 wait() 方法，线程挂起，不占用 CPU，直到被唤醒
 * 2. 获取优化目标 (Get Data)：从 Map 中获取“活跃”的关键帧和地图点
 * 3. 执行优化 (Optimize)：调用 Optimize() 方法，构建并求解优化问题
 * 
 * 关键帧选取策略：如果某一帧追踪到前一帧的关键点数量很少，说明视角变化较大或场景复杂，适合做关键帧
 * 滑动窗口大小：通常设为7帧，是一个经验值，既能保证约束充足，又不会计算过慢
 * 
 * 激活的路标点选取策略：只保留那些被“激活关键帧”观测到的路标点
 */
void Backend::BackendLoop() {
    // 只要系统还在运行，就一直循环
    while (backend_running_.load()) {
        // 1. 等待信号 (Wait)
        {   // 作用域块：控制锁的生命周期
            std::unique_lock<std::mutex> lock(data_mutex_);
            // 线程挂起（睡眠），不占用 CPU 资源。
            // 只有当 UpdateMap() 或 Stop() 调用 notify_one() 时，它才会被唤醒。
            map_update_.wait(lock);
        }

        // 后端仅优化激活的Frames和Landmarks
        // 2. 获取优化目标 (Get Data)
        // 醒来后，从 Map 中获取“活跃”的关键帧和地图点。
        // SLAM 设计哲学：滑动窗口优化 (Sliding Window Optimization)。
        // 我们只优化“最近的 7 帧 (Active Keyframes) 和它们看到的点”，
        // 而不是优化整个地图的所有点，从而保证计算量是常数级的 (O(1))，满足实时性。
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        // 3. 执行优化 (Optimize)
        Optimize(active_kfs, active_landmarks);
    }
}

/**
 * Optimize - 核心优化函数
 * 使用 G2O 库构建并求解一个 BA（Bundle Adjustment）优化问题
 * 输入：一组关键帧和路标点
 * 输出：优化后的关键帧位姿和路标点位置，直接更新回 Map 中
 */
void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {
    // 1. 定义求解器类型
    // BlockSolver_6_3: 优化变量分两块，相机位姿是 6 维 (SE3)，路标点是 3 维 (XYZ)。
    // 这是 BA 问题的标准配置。
    typedef g2o::BlockSolver_6_3 BlockSolverType;

    // 2. 线性求解器 (Linear Solver)
    // LinearSolverCSparse: 使用 CSparse 库求解稀疏矩阵 (H * dx = -b)。
    // BA 的 H 矩阵（Hessian）具有特殊的“箭头形”稀疏结构，CSparse 能高效利用这一特性（舒尔补）。
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
        LinearSolverType;

    // 3. 算法选择
    // OptimizationAlgorithmLevenberg: 使用 LM 算法，它是最稳健的非线性优化算法。
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(
            std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // pose 顶点，使用Keyframe id，因为要优化的位姿和关键帧是一一对应的
    // 辅助容器：记录 ID 到 顶点指针 的映射，方便后面建边
    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;        // 用于给地图点生成唯一 ID (防止和 KeyFrame ID 冲突)

    // --- 添加位姿顶点 (KeyFrames) ---
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;

        // 创建自定义的 VertexPose 顶点 (定义在 g2o_types.h)
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(kf->keyframe_id_);   // 设 ID
        vertex_pose->setEstimate(kf->Pose());   // 设初值 (Initial Guess)
        optimizer.addVertex(vertex_pose);

        // 记录最大关键帧 ID，用于后续生成地图点 ID
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // --- 准备添加路标顶点 ---
    // (路标顶点在遍历边的时候按需添加，见下文)
    // 路标顶点，使用路标id索引
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

    // 获取相机参数 (用于构建边)
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();

    /**
     * 添加边 (Edges)，这是最繁琐的部分。我们需要遍历每个地图点，找到它被哪些关键帧观测到了，然后建立连接
     */
    int index = 1;
    double chi2_th = 5.991;  // robust kernel 阈值，对应95%的置信度（自由度2）卡方阈值 (用于 Huber 核)
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    // 遍历所有活跃地图点
    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_) continue;     // 已经是外点的不要
        unsigned long landmark_id = landmark.second->id_;

        // 获取该点的观测列表 (Observations)
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations) {
            if (obs.lock() == nullptr) continue;        // 特征点可能已被销毁
            auto feat = obs.lock();

            // 检查特征点本身是否是外点，以及对应的帧是否存在
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

            auto frame = feat->frame_.lock();

            // --- 创建边 (Edge) ---
            EdgeProjection *edge = nullptr;
            // 根据是左眼还是右眼，传入不同的外参 (Extrinsics)
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K, right_ext);
            }

            // --- 懒加载：添加路标顶点 ---
            // 如果这个路标点还没加到图里，现在加进去
            if (vertices_landmarks.find(landmark_id) == vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->Pos());     // 初值：当前的 3D 坐标
                v->setId(landmark_id + max_kf_id + 1);  // ID 生成策略：为了不和 KeyFrame ID 重复，加上 max_kf_id
                
                // 【关键知识点】setMarginalized(true)
                // 告诉 G2O：求解线性方程时，先消元这个点 (Schur Complement)。
                // BA 中路标点极多，位姿点少。先消掉路标点，可以让矩阵大大减小，加速求解。
                v->setMarginalized(true);

                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            // --- 连接边 ---
            // 只有当这条边的两头（帧和点）都在优化图里时，才连线
            // (注意：后端只优化 active_keyframes，但一个点可能被非 active 的帧观测到，那种观测我们要忽略)
            if (vertices.find(frame->keyframe_id_) != vertices.end() && 
                vertices_landmarks.find(landmark_id) != vertices_landmarks.end()) {
                    edge->setId(index);
                    // 连接两个顶点：0号是 Pose，1号是 Landmark
                    edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
                    edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
                    edge->setMeasurement(toVec2(feat->position_.pt));   // 观测值：像素坐标
                    edge->setInformation(Mat22::Identity());            // 信息矩阵 (权重)
                    auto rk = new g2o::RobustKernelHuber();         // 鲁棒核函数 (Huber)
                    rk->setDelta(chi2_th);
                    edge->setRobustKernel(rk);

                    edges_and_features.insert({edge, feat});
                    optimizer.addEdge(edge);
                    index++;
                }
            else delete edge;   // 没连上就删掉，防止内存泄露
                
        }
    }

    // do optimization and eliminate the outliers
    // 1. 初次优化
    optimizer.initializeOptimization();
    optimizer.optimize(10);     // 迭代 10 次

    // 2. 自适应阈值调整循环
    // 如果优化完发现内点太少 (<50%)，说明初始误差太大，Huber 核把太多点当成外点了。
    // 策略：放大阈值 (chi2_th *= 2)，再试一次。 最多试 5 次。
    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        // 统计内点率
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {   
            break;              // 内点率 > 50%，认为结果可信，跳出
        } else {
            chi2_th *= 2;       // 放宽标准
            iteration++;
        }
    }

    // 3. 处理外点
    // 根据最终的阈值，把误差太大的边对应的 Feature 标记为 outlier。
    // 并且从 MapPoint 中移除这个观测。
    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // Set pose and lanrmark position
    // 4. 写回结果 (Write Back)
    // 优化后的结果还在 G2O 的顶点里，需要拷回到 Frame 和 MapPoint 对象中。
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace myslam