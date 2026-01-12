cmake_minimum_required(VERSION 2.8)     # 声明构建该项目所需的 CMake 最低版本是 2.8。如果用户安装的版本低于此版本，CMake 会报错停止
project(myslam) # 定义项目名称为 myslam。这会定义一些变量，比如 PROJECT_SOURCE_DIR（项目根目录）和 PROJECT_BINARY_DIR（构建目录）

set(CMAKE_BUILD_TYPE Release)   # 设置构建类型为 Release。这意味着编译器会开启优化（如 -O3），并且移除调试信息，使程序运行速度最快。调试时通常改为 Debug

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall")         # 设置 C++ 标准为 C++17，并开启所有警告
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++17 -O3 -fopenmp -pthread")        # 针对 Release 模式追加特定的优化标志

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)      # 告诉 CMake 去哪里找第三方库的查找脚本（FindXXX.cmake）。这里指定了项目根目录下的 cmake_modules 文件夹
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)   # 生成的可执行文件（如 run_kitti_stereo）会放在根目录的 bin/ 下
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)      # 生成的库文件（如 libmyslam.so）会放在根目录的 lib/ 下

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")      # 手动添加 Eigen 库的头文件路径。虽然通常建议用 find_package(Eigen3)，但直接指定路径也是一种简单粗暴的方法

# OpenCV
find_package(OpenCV 4.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})     # 查找 OpenCV 库，要求版本至少为 4.1。REQUIRED 表示找不到就报错停止。找到后将 OpenCV 的头文件路径加入编译包含路径

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})   # 查找 Pangolin 库（用于可视化 GUI），并包含其头文件

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})     # 查找 Sophus 库（李代数库），并包含其头文件

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})        # 查找 G2O 库（图优化库）。这里会用到之前设置的 cmake_modules 中的 FindG2O.cmake 脚本

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})       # 查找 Google Log 库（日志系统）

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})      # 查找 Google Test 库（单元测试框架）

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})     # 查找 GFlags 库（命令行参数解析）

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})     # 查找 CSparse 库（稀疏矩阵运算库，G2O 的依赖项）

# 定义一个变量 THIRD_PARTY_LIBS，把所有找到的库文件（.so 或 .a）汇总在一起
# 这样在后续 target_link_libraries 时，只需要引用这个变量，不用把几十个库名再写一遍
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        )

# 启用 CMake 的测试功能（配合 CTest 使用）
enable_testing()

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)      # 将项目自己的头文件目录（include/）加入包含路径。这样代码里写 #include "myslam/frame.h" 才能找到文件
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)
