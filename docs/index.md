# 视觉 SLAM 十四讲 · 学习与实践

> **从理论到代码，构建完整的 SLAM 知识体系。**
>
> *Visual SLAM: From Theory to Practice - A Learning Journey*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/gaoxiang12/slambook2)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-14%2F17-blue.svg)](https://isocpp.org/)
[![SLAM](https://img.shields.io/badge/SLAM-Visual_Odometry-orange)](https://github.com/gongzihang6/slambook2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目简介

欢迎来到我的 **视觉 SLAM 十四讲（第二版）** 学习仓库。

本项目基于高翔博士的经典著作《视觉 SLAM 十四讲》，旨在记录学习过程中的**代码实现**、**习题解答**以及**工程实践笔记**。这不仅是一个代码备份，更是一份深度解析 SLAM 算法与工程落地的知识手册。

### 核心内容

* **📘 源码剖析**：对书中每一讲的示例代码进行复现，并添加详细的中文注释，解释算法背后的数学原理（李群李代数、非线性优化等）。
* **🛠️ 工程实战**：整理了 SLAM 开发中必备的工具链使用指南（CMake, g++, Eigen, Sophus, OpenCV, PCL, g2o, Ceres 等）。
* **📝 习题详解**：提供课后习题的详细推导与解答。
* **🐛 踩坑记录**：记录在不同环境（Ubuntu/Mac）下的编译报错与解决方案。

---

## 🗺️ 学习路线与进度

目前涵盖章节与进度如下：

| 章节 | 主题 | 状态 | 关键词 |
| :--- | :--- | :---: | :--- |
| **Ch 2** | 初识 SLAM | ✅ 完成 | `CMake` `g++` `HelloSLAM` |
| **Ch 3** | 三维空间刚体运动 | ✅ 完成 | `Eigen` `旋转矩阵` `四元数` |
| **Ch 4** | 李群与李代数 | ✅ 完成 | `Sophus` `李代数求导` |
| **Ch 5** | 相机与图像 | ✅ 完成 | `OpenCV` `点云拼接` `畸变矫正` |
| **Ch 6** | 非线性优化 | ✅ 完成 | `Gauss-Newton` `Ceres` `g2o` |
| **Ch 7** | 视觉里程计 (1) | ✅ 完成 | `ORB` `特征匹配` `3D-2D` `PnP` |
| **Ch 8** | 视觉里程计 (2) | ✅ 完成 | `光流法` `直接法` |
| **Ch 9** | 后端 1 | ✅ 完成 | `Bundle Adjustment` |
| **Ch 10** | 后端 2 | 🚧 进行中 | `位姿图` `因子图` |
| **...** | 回环检测与建图 | ⏳ 待定 | `BoW` `Octomap` |

---

## 🚀 工程效能指南 (Featured)

在学习初期，环境配置和编译构建往往是最大的拦路虎。我特别整理了 **CMake 与 g++ 的现代化使用指南**，帮助快速上手 C++ 工程构建。

### 💡 CMake 核心速查

!!! tip "拒绝硬编码，拥抱现代化 CMake"
    不要再手动写绝对路径了！使用 `find_package` 和 `target_...` 指令集来管理你的依赖。

* **寻找依赖**: `find_package(PCL REQUIRED)` — 让 CMake 自动帮你找库。
* **链接库**: `target_link_libraries(my_app ${PCL_LIBRARIES})` — 核心指令。
* **头文件**: `target_include_directories(my_app PRIVATE ...)` — 避免污染全局命名空间。

👉 **[点击查看完整的 CMake & g++ 编译指令详解](docs/ch2/README.md)** *(包含 `-I`, `-L` 与 CMake 指令的对应关系表)*

---

## 📂 仓库结构

```text
slambook2/
├── ch2/ ~ ch13/        # 各章节源代码 (Source Code)
├── docs/               # 学习笔记文档 (Documentation)
├── figures/            # 文档图片资源
├── mkdocs.yml          # 文档站点配置文件
└── README.md           # 项目主页