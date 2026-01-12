# `cv::calcOpticalFlowPyrLK` 函数详解

`cv::calcOpticalFlowPyrLK` 函数是 `OpenCV` 中实现金字塔 Lucas-Kanade 光流法的核心函数。核心原理如下：1、Lucas-Kanade（LK）算法基于 3 个假设（亮度恒定、小运动、空间一致性），通过求解线性方程组，计算像素在两张图像间的位移向量(dx, dy)；2、金字塔（Pyramids）是为了解决“小运动”假设的限制（如果物体运动太快或者幅度太大，单纯 LK 算不准），算法会先把图像缩小建立金字塔。现在顶层（低分辨率）算一个粗略的位移，作为下一层（高分辨率）的初值（因为原始分辨率的大运动，在缩小后的低分辨率图像中就变小了，LK 估计更准），层层下推，从而能够追踪较大的像素移动。

函数详细解析

```c++
cv::calcOpticalFlowPyrLK(
    current_frame_->left_img_,   // [输入] prevImg: 上一时刻图像（这里指左图）
    current_frame_->right_img_,  // [输入] nextImg: 当前时刻图像（这里指右图）
    kps_left,               // [输入] prevPts: 左图中的特征点坐标 (vector<Point2f>)
    kps_right,             // [输入/输出] nextPts: 右图中对应的特征点坐标
    status,               // [输出] status: 每一对点是否追踪成功 (1=成功, 0=失败)
    error,                // [输出] err: 对应点的误差估计
    cv::Size(11, 11),        // [输入] winSize: 搜索窗口大小
    3,                   // [输入] maxLevel: 金字塔层数
    cv::TermCriteria(...),       // [输入] criteria: 迭代终止条件
    cv::OPTFLOW_USE_INITIAL_FLOW // [输入] flags: 标志位 (关键！)
);
```

1、`current_frame_->left_img_` / `right_img_`：输入必须是单通道灰度图（`CV_8UC1`），这里使用左图作为前一帧，右图作为后一帧，在空间上进行追踪；

2、`kps_left`：左图（前一帧）中的特征点像素坐标；

3、`kps_right`：这是输出结果，存放计算出的右图点坐标。因为使用了 `OPTFLOW_USE_INITIAL_FLOW` 标志，这个变量同时也是输入。因为调用代码前，我们已经给 `kps_right` 提供了初始值（比如假设右图点和左图点坐标一样，或者根据地图点投影得到）。

4、`status`：`std::vector<uchar>` 类型。长度与输入点数一致，如果第 $i$ 个点追踪成功，`status[i] = 1`；如果跑出图像边界或无法收敛，`status[i] = 0`，用于输出光流跟踪结果；

5、`cv::Size(11, 11)`：光流窗口大小。窗口越大，对噪声越鲁棒，但对快速变化的运动越不敏感；窗口越小，定位越准，但容易受噪声干扰。$11 \times 11$ 是经验值；

6、`3` (`maxLevel`)：金字塔层数。0 表示只用原图。3 表示共 4 层（原图 + 3 层缩放）。这允许算法处理较大的视差（Disparity）

7、`cv::TermCriteria(...)`：`cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01`，这是指定了迭代终止条件，要么迭代了 30 次，要么某次迭代的搜索移动距离小于 0.01 像素，就停止计算；

8、`cv::OPTFLOW_USE_INITIAL_FLOW` (核心标志位)：告诉 OpenCV，“不要从原点开始搜，请用我传入的 `kps_right` 中的坐标作为搜索的起始位置”。如果不加这个标志，LK 算法会假设初始位移为 0（即右图点就在左图点原来的位置）。但在双目模式下，我们可能已经知道了一些先验信息（比如以前的地图点投影），提供初值可以 **极大提高收敛速度和匹配成功率**；

