#ifndef MYSLAM_DATASET_H
#define MYSLAM_DATASET_H    
#include "myslam/camera.h"      // Dataset 需要提供 Camera（相机内参），所以引用 camera.h
#include "myslam/common_include.h"
#include "myslam/frame.h"       // Dataset 需要生产 Frame（下一帧图像），所以引用 frame.h

namespace myslam {

/**
 * 数据集读取
 * 构造时传入配置文件路径，配置文件的dataset_dir为数据集路径
 * Init之后可获得相机和下一帧图像
 * 
 * 单独写一个数据读取类，实现核心算法VO与数据来源的解耦，**核心算法（VO）**不应该关心数据是从哪里来的
 */
class Dataset {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;

    // 构造函数:仅仅记录数据集的路径，不做耗时操作
    Dataset(const std::string& dataset_path);

    /**
     * 初始化，返回是否成功
     * 这是一个重量级函数。它会去检查路径是否存在，读取 calib.txt 解析相机内参，预加载文件名列表等
     * 将其与构造函数分开是为了更好地处理错误。如果文件不存在，Init 返回 false，主程序可以优雅地报错退出；如果在构造函数里失败，处理异常会比较麻烦
     */
    bool Init();

    /**
     * 读取磁盘上的下一张图片，打包成 Frame 对象返回
     * SLAM 是一个时间序列过程。系统就像一条流水线，Dataset 是流水线的源头（Source）
     * 每次调用 NextFrame，内部的游标（current_image_index_）就会加 1
     * 返回 Frame::Ptr 而不是 cv::Mat，是因为 Frame 包含的信息更多（ID、时间戳、相机模型等），是系统通用的数据包
     */
    Frame::Ptr NextFrame();

    // get camera by id
    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

private:
    std::string dataset_path_;
    int current_image_index_ = 0;

    std::vector<Camera::Ptr> cameras_;
};
}  // namespace myslam

#endif