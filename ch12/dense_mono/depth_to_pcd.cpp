/**
 * depth_to_pcd.cpp
 * 功能：读取深度图（XML/YML格式 或 16位PNG）和 RGB图像，生成带颜色的 3D 点云
 */

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;
using namespace cv;

// ------------------------------------------------------------------
// 相机内参 (直接复制自 dense_mapping.cpp)
const double fx = 481.2f;
const double fy = -480.0f; // 注意这里是负值
const double cx = 319.5f;
const double cy = 239.5f;
// ------------------------------------------------------------------

// ./depth_to_pcd depth.xml 
int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Usage: ./depth_to_pcd <depth_file_path> [rgb_image_path]" << endl;
        cout << "Example: ./depth_to_pcd depth.xml ../images/scene_000.png" << endl;
        return -1;
    }

    string depth_path = argv[1];
    string rgb_path = (argc > 2) ? argv[2] : "";

    // 1. 读取深度图
    Mat depth;
    // 检查文件扩展名，决定读取方式
    if (depth_path.find(".xml") != string::npos || depth_path.find(".yml") != string::npos) {
        // 如果是 XML/YML (推荐，精度最高)
        FileStorage fs(depth_path, FileStorage::READ);
        fs["depth"] >> depth;
        fs.release();
    } else {
        // 如果是普通图片 (假设是 16位 unsigned short 或 32位 float)
        depth = imread(depth_path, IMREAD_UNCHANGED);
    }

    if (depth.empty()) {
        cerr << "Failed to load depth map: " << depth_path << endl;
        return -1;
    }

    // 2. 读取 RGB 图像 (可选)
    Mat rgb;
    if (!rgb_path.empty()) {
        rgb = imread(rgb_path);
        if (rgb.empty()) {
            cout << "Warning: Could not load RGB image, generating colorless point cloud." << endl;
        }
    }

    // 3. 定义点云类型
    // 如果有 RGB 图，用 PointXYZRGB，否则用 PointXYZ
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 4. 遍历像素，反投影
    // 假设 depth 图是 CV_64F (double)，单位为米
    // 如果是 CV_16U，通常单位是毫米，需要除以 1000.0
    
    // 4. 遍历像素，反投影 (修正版)
    for (int v = 0; v < depth.rows; v++) {
        for (int u = 0; u < depth.cols; u++) {
            double d_ray = 0.0; // 注意：这里读出来的是射线长度(欧氏距离)
            
            // 读取深度值
            if (depth.type() == CV_64F) {
                d_ray = depth.at<double>(v, u);
            } else if (depth.type() == CV_32F) {
                d_ray = depth.at<float>(v, u);
            }
            
            // 过滤无效点
            if (d_ray <= 0.01 || d_ray >= 10.0) continue;

            // 【核心修正逻辑】
            // 1. 计算未归一化的方向向量 (假设 z=1)
            double x_norm = (u - cx) / fx;
            double y_norm = (v - cy) / fy;
            double z_norm = 1.0;

            // 2. 计算该方向向量的模长 (为了归一化)
            double norm_inv = sqrt(x_norm*x_norm + y_norm*y_norm + z_norm*z_norm);
            
            // 3. 计算真实的 3D 坐标
            // 原理： Point = Direction_Normalized * Depth_Ray
            // Direction_Normalized = (x, y, 1) / norm_inv
            pcl::PointXYZRGB p;
            p.x = (x_norm / norm_inv) * d_ray;
            p.y = (y_norm / norm_inv) * d_ray;
            p.z = (z_norm / norm_inv) * d_ray; 

            // 赋予颜色 (同前)
            if (!rgb.empty()) {
                Vec3b color = rgb.at<Vec3b>(v, u);
                p.b = color[0]; p.g = color[1]; p.r = color[2];
            } else {
                p.b = 255; p.g = 255; p.r = 255;
            }

            cloud->points.push_back(p);
        }
    }

    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    cout << "Point cloud generated with " << cloud->points.size() << " points." << endl;

    // 5. 保存
    string output_name = "reconstructed_cloud.pcd";
    pcl::io::savePCDFileBinary(output_name, *cloud);
    cout << "Saved to " << output_name << endl;

    return 0;
}
