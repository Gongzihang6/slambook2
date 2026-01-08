cmake_minimum_required(VERSION 2.8)
project(dense_monocular)

add_definitions(-DBOOST_TIMER_ENABLE_DEPRECATED)        # 这行代码的作用是定义一个宏，强制 Boost 允许使用旧版 Timer 头文件

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -march=native -O3")

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 4.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# 寻找 PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 1. 寻找 OpenMP 包 (新增)
find_package(OpenMP REQUIRED)

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES})

add_executable(dense_mapping dense_mapping.cpp)
target_link_libraries(dense_mapping 
        ${THIRD_PARTY_LIBS}
        OpenMP::OpenMP_CXX      # 链接 OpenMP 库 (新增，注意这行的写法)
)

# 编译可执行文件
add_executable(depth_to_pcd depth_to_pcd.cpp)

# 链接库
target_link_libraries(depth_to_pcd ${OpenCV_LIBS} ${PCL_LIBRARIES})