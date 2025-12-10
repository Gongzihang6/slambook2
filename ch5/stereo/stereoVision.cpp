#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;

// 文件路径
string left_file = "../stereo/left.png";
string right_file = "../stereo/right.png";

// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char **argv) {

    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线
    double b = 0.573;

    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);    // 参数0表示读取为灰度图
    cv::Mat right = cv::imread(right_file, 0);

    /*
    static cv::Ptr<...> cv::StereoSGBM::create(
        int minDisparity = 0,       // 最小视差值。通常为0
        int numDisparities = 16,    // 视差搜索范围。即算法会在当前像素u的左侧[u - minDisp, u - minDisp - numDisp]范围内寻找匹配点
        int blockSize = 3,          // 匹配块大小
        int P1 = 0,                 // 惩罚系数 1。控制视差图的平滑度。
        int P2 = 0,                 // 惩罚系数 2。对应视差变化大于 1 的像素产生的惩罚。
        int disp12MaxDiff = 0,      // 左右一致性检查 (Left-Right Check) 允许的最大差异
        int preFilterCap = 0,       // 预处理滤波器的截断值。用于降低光照变化的影响。
        int uniquenessRatio = 0,    // 唯一性比率。最佳匹配代价必须比次佳匹配代价好至少 10%。
        int speckleWindowSize = 0,  // 散斑过滤窗口。如果一个视差连通区域的像素数量小于 100，则被视为噪点（Speckle）并剔除。
        int speckleRange = 0,       // 定义什么样的视差差异算作“同一个连通区域”。
        int mode = 0)
    */
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    /*
    compute 输出的 disparity_sgbm 是 CV_16S（16位有符号整数）。
    关键点：OpenCV 为了保持精度，输出的视差值是 定点数，数值被放大了 16 倍。
    因此，convertTo 时必须乘以 1.0/16.0f 才能还原为真实的像素级视差。
    */
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色

            // 根据双目模型计算 point 的位置
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);
        }

    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    // 画出点云
    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    // 创建一个标题为 "Point Cloud Viewer" 的窗口，宽1024像素，高768像素
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);

    // 【关键】开启深度测试。
    // 如果不开启，后画的点会覆盖先画的点，导致你透过前面的墙看到后面的东西。
    // 开启后，GPU会根据 Z 轴距离判断遮挡关系。
    glEnable(GL_DEPTH_TEST);

    // 开启混合模式（通常用于处理透明度），这里其实画不透明点云非必须，但在SLAM通用代码中很常见
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 定义一个渲染状态对象 s_cam
    pangolin::OpenGlRenderState s_cam(
        // 1. 投影矩阵 (ProjectionMatrix)：定义相机的内参（模拟人眼或相机镜头）
        // 参数依次为：窗口宽 w, 窗口高 h, 焦距 fu, 焦距 fv, 光心 u0, 光心 v0, 最近可视距离 near, 最远可视距离 far
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),

        // 2. 模型视图矩阵 (ModelViewLookAt)：定义相机在虚拟世界里的位置（外参）
        // 前三个参数 (0, -0.1, -1.8)：相机所在的位置 (Camera Position)
        // 中间三个参数 (0, 0, 0)：相机看向的目标点 (Look At Target)
        // 最后三个参数 (0.0, -1.0, 0.0)：相机的“上方”指向哪里 (Up Vector)，这里y轴向下，符合CV习惯
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // 创建一个视图 d_cam，用来显示 3D 内容
    pangolin::View &d_cam = pangolin::CreateDisplay()
        // SetBounds 设置视图在窗口中的范围 (0.0 到 1.0 表示占满整个窗口)
        // 第三个参数 pangolin::Attach::Pix(175) 通常用于给左侧留出控制面板的空间（虽然这里没画面板）
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        // 【关键】绑定交互句柄。
        // 这行代码让你能用鼠标左键旋转、右键缩放、中键平移。
        .SetHandler(new pangolin::Handler3D(s_cam));

    // 只要用户没按 ESC 或关闭窗口，就一直循环
    while (pangolin::ShouldQuit() == false) {
        // 1. 清除上一帧的画面（颜色缓存）和深度信息（深度缓存）
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 2. 激活 d_cam 视图，并应用刚才定义的相机状态 s_cam
        // 这会将 OpenGL 的矩阵设置为我们定义的视角
        d_cam.Activate(s_cam);
        // 3. 设置背景颜色为白色 (R=1, G=1, B=1)
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // 4. 开始画点
        glPointSize(2);     // 设置点的大小为 2 像素
        glBegin(GL_POINTS); // 告诉 OpenGL：接下来给你的坐标，都当成“点”来画
        for (auto &p: pointcloud) {
            // 设置当前点的颜色。p[3] 是归一化的灰度值 (0~1)
            // glColor3f(R, G, B)，这里三个分量一样，所以是灰色/黑白
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);   // 设置当前点的坐标 (x, y, z)
        }
        glEnd();

        // 在 glEnd() 之后添加
        d_cam.Activate(s_cam); // 确保在正确的矩阵下绘制
        glLineWidth(3);
        glBegin(GL_LINES);
        // X轴 - 红
        glColor3f(1.0, 0.0, 0.0); glVertex3d(0,0,0); glVertex3d(0.5, 0, 0);
        // Y轴 - 绿
        glColor3f(0.0, 1.0, 0.0); glVertex3d(0,0,0); glVertex3d(0, 0.5, 0);
        // Z轴 - 蓝
        glColor3f(0.0, 0.0, 1.0); glVertex3d(0,0,0); glVertex3d(0, 0, 0.5);
        glEnd();

        // 5. 交换缓冲区，显示这一帧
        pangolin::FinishFrame();

        // 6. 稍微睡一下，防止这个空循环占用 100% CPU
        usleep(5000);   // sleep 5 ms
    }
    return;
}