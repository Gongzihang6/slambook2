#include "myslam/dataset.h"
#include "myslam/frame.h"

#include <boost/format.hpp>     // 用于字符串格式化，类似 printf 但用于 C++ string
#include <fstream>              // 文件流操作
#include <opencv2/opencv.hpp>   // OpenCV 图像处理
using namespace std;

namespace myslam {

// 简单的构造函数，接收数据集路径作为参数，初始化数据集路径
Dataset::Dataset(const std::string& dataset_path): dataset_path_(dataset_path) {}

// 负责解析 KITTI 的 calib.txt 文件
bool Dataset::Init() {
    // 1. 打开标定文件
    // 拼接路径，打开 calib.txt
    ifstream fin(dataset_path_ + "/calib.txt");
    if (!fin) {
        // 如果打不开（路径错误或文件不存在），输出错误日志并返回失败
        LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
        return false;
    }

    // 2. 循环读取 4 个相机的参数
    // KITTI 数据集通常包含 4 个相机：P0, P1 (灰度), P2, P3 (彩色)
    for (int i = 0; i < 4; ++i) {
        // 读取每行的头部，例如 "P0: "
        char camera_name[3];
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }

        // 3. 读取 12 个投影参数
        // KITTI 的标定文件给出的是 3x4 的投影矩阵 P
        // P = K * [R | t] (对于校正后的图像，R 通常为单位阵)
        // P = [fx, 0, cx, tx]
        //     [0, fy, cy, ty]
        //     [0,  0,  1, tz]
        double projection_data[12];
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }

        // 4. 提取内参矩阵 K (3x3)
        Mat33 K;
        K << projection_data[0], projection_data[1], projection_data[2],
            projection_data[4], projection_data[5], projection_data[6],
            projection_data[8], projection_data[9], projection_data[10];

        // 5. 提取平移部分 t (实际上是 K * t)
        // 投影矩阵的第 4 列不仅仅是平移向量 t，而是 K * t
        Vec3 t;
        t << projection_data[3], projection_data[7], projection_data[11];

        // 6. 恢复真实的平移向量 t
        // 公式：P_col4 = K * t  =>  t = K_inv * P_col4
        // 对于 P1 (右眼相机)，这里算出来的 t 的模长就是双目基线 (Baseline)
        t = K.inverse() * t;

        // 7. 对内参进行缩放 (关键步骤！)
        // 为什么乘 0.5？因为在 NextFrame 函数中，图像被 cv::resize 缩小了一半。
        // 图像缩小一半，焦距 fx, fy 和主点 cx, cy 也要相应缩小一半，否则几何关系就错了。
        K = K * 0.5;

        // 8. 创建 Camera 对象并存入列表
        // 构造函数参数：fx, fy, cx, cy, 基线长度, 位姿 SE3
        // SE3(SO3(), t) 表示旋转为单位阵（因为是校正图像），平移为刚才算出的 t
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), SE3(SO3(), t)));
        cameras_.push_back(new_camera);
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    fin.close();
    current_image_index_ = 0;   // 重置图片索引，从第 0 张开始读
    return true;
}

Frame::Ptr Dataset::NextFrame() {
    // 1. 定义文件名格式
    // boost::format 用于生成类似 "dataset_dir/image_0/000000.png" 的字符串
    // %s 对应 dataset_path_
    // %d 对应相机编号 (0 或 1)
    // %06d 对应 current_image_index_ (如 0 变成 000000)
    boost::format fmt("%s/image_%d/%06d.png");
    cv::Mat image_left, image_right;

    // 2. 读取左目 (image_0) 和右目 (image_1) 图像
    // str() 将 format 对象转换为 std::string
    // cv::IMREAD_GRAYSCALE: 以灰度模式读取，VO 通常不需要彩色信息
    image_left =
        cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);

    // 3. 检查是否读完
    // 如果读取失败（data 为空），说明可能已经读到了数据集的末尾
    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(WARNING) << "cannot find images at index " << current_image_index_;
        return nullptr;
    }

    // 4. 图像缩放 (Downsampling)
    // 为了提高运行速度，将图像长宽各缩小为原来的 0.5 倍
    // 这与 Init() 中 K = K * 0.5 是对应的
    cv::Mat image_left_resized, image_right_resized;
    cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);

    // 5. 创建 Frame 对象
    // 使用工厂函数创建新帧，返回帧对象的智能指针
    auto new_frame = Frame::CreateFrame();

    // 6. 填充数据
    // 将处理后的图像赋值给 frame 的成员变量
    new_frame->left_img_ = image_left_resized;
    new_frame->right_img_ = image_right_resized;

    // 7. 更新索引
    current_image_index_++;
    return new_frame;
}

}  // namespace myslam