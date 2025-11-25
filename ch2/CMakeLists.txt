# 声明要求的 cmake 最低版本
cmake_minimum_required(VERSION 2.8)

# 声明一个 cmake 工程
project(HelloSLAM)

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
# add_executable(helloSLAM helloSLAM.cpp)
add_executable(helloSLAM helloSLAM2.cpp)

add_library( hello libHelloSLAM.cpp )

# 生成一个共享库（SHARED关键字），这样在链接时可以使用动态链接，运行时才会被加载
# 语法：add_library( 库名 SHARED 源代码文件 ），项目内的可执行文件都可以链接这个库，inclide头文件后就可以使用里面的方法
# 这是动态链接，在运行时才会被加载，所以生成的可执行文件体积会比较小
# 静态链接是在编译时就将库的代码静态链接到可执行文件中，所以生成的可执行文件体积会比较大
add_library( hello_shared SHARED libHelloSLAM.cpp )

add_executable( useHello useHello.cpp )
target_link_libraries( useHello hello_shared )  # 将 useHello 链接到 hello_shared 库

set(CMAKE_BUILD_TYPE "Debug")      # 若设置为"Release"模式则改为"Debug"，否则断点设置无效