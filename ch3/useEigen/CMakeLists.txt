cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# [关键步骤] 生成编译命令数据库
# 这会生成一个 compile_commands.json 文件，VS Code/Trae 能够自动读取它
# 从而知道头文件在哪里，消除红色波浪线
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 添加Eigen头文件
include_directories("/usr/include/eigen3")
add_executable(eigenMatrix eigenMatrix.cpp)
add_executable(linearEqSolution linearEqSolution.cpp)
add_executable(matrixExtractAssign matrixExtractAssign.cpp)
