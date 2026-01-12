//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

/**
 * 定义变量：gflags 库会自动生成一个名为 FLAGS_config_file 的全局字符串变量
 * 默认值：如果你在命令行不传参数，它的值就是 "./config/default.yaml"
 * 解析：google::ParseCommandLineFlags 会解析命令行输入（例如 ./run_kitti_stereo --config_file=./my_config.yaml），并覆盖 FLAGS_config_file 的值
 */
DEFINE_string(config_file, "./config/default.yaml", "config file path");

int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}
