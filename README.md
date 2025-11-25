# 视觉 SLAM 十四讲学习历程

## 2025-11-25

fork 作者的代码 https://github.com/gaoxiang12/slambook2，看完书籍的第二讲；

### 了解 g++编译命令和 CMake 指令

我将它们分成了五大类，并区分了 **“老式/全局写法”**（影响所有文件）和 **“现代/目标级写法”**（推荐，只影响特定程序）。

#### 1. 头文件与库路径 (-I, -L)

| **g++ 参数**                        | **CMake 指令 (对应关系)**                                    | **示例**                                                     |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **`-I<path>`** (指定头文件路径)     | **`include_directories(...)`** (全局) **`target_include_directories(...)`** (推荐) | `g++ -I/usr/local/include` ⬇ `target_include_directories(my_app PRIVATE /usr/local/include)` |
| **`-L<path>`** (指定库文件搜索路径) | **`link_directories(...)`** *(注: CMake 中常用 find_package 自动处理，很少手动写这个)* | `g++ -L/usr/local/lib` ⬇ `link_directories(/usr/local/lib)`  |

#### 2. 库链接 (-l)

| **g++ 参数**                  | **CMake 指令 (对应关系)**                                    | **示例**                                                     |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **`-l<name>`** (链接具体的库) | **`target_link_libraries(...)`** *(这是 CMake 中最核心的指令)* | `g++ -lpcl_common` ⬇ `target_link_libraries(my_app pcl_common)` |

#### 3. 编译选项与宏 (-g, -O, -D, -Wall)

| **g++ 参数**                       | **CMake 指令 (对应关系)**                                    | **示例**                                                     |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **`-D<Macro>`** (定义宏)           | **`add_definitions(...)`** (全局) **`target_compile_definitions(...)`** (推荐) | `g++ -DDEBUG` ⬇ `target_compile_definitions(my_app PRIVATE DEBUG)` |
| **`-Wall` / `-O3`** (通用编译参数) | **`add_compile_options(...)`** (全局) **`target_compile_options(...)`** (推荐) | `g++ -Wall` ⬇ `add_compile_options(-Wall)`                   |
| **`-g` / `-O`** (构建类型快捷方式) | **`set(CMAKE_BUILD_TYPE ...)`** *(这是 CMake 特有的高级抽象)* | `g++ -g` ⬇ `set(CMAKE_BUILD_TYPE "Debug")`  `g++ -O3` ⬇ `set(CMAKE_BUILD_TYPE "Release")` |

#### 4. 标准与输出 (-std, -o)

| **g++ 参数**                     | **CMake 指令 (对应关系)**            | **示例**                                                     |
| -------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **`-std=c++17`** (C++标准)       | **`set(CMAKE_CXX_STANDARD 17)`**     | `g++ -std=c++17` ⬇ `set(CMAKE_CXX_STANDARD 17)`              |
| **`-o <name>`** (生成可执行文件) | **`add_executable(<name> ...)`**     | `g++ ... -o my_app` ⬇ `add_executable(my_app main.cpp)`      |
| **`-shared -fPIC`** (生成动态库) | **`add_library(<name> SHARED ...)`** | `g++ -shared -fPIC ... -o libmy.so` ⬇ `add_library(my SHARED lib.cpp)` |



### CMake常用命令指南

`find_package`命令

#### 1. 为什么要用它？

-   笨办法（硬编码）：

    include_directories(/usr/include/pcl-1.12)

    -   *坏处：* 你的电脑装的是 PCL 1.12，导师的服务器装的是 PCL 1.10，路径变了，代码直接报错。你需要手动去改 `CMakeLists.txt`。

-   聪明办法（find_package）：

    find_package(PCL REQUIRED)

    -   *好处：* CMake 会自动去系统的标准路径里“搜寻”PCL 的配置文件。不管它在 `/usr/include` 还是 `/opt/local`，只要找到了，它就会把路径自动填到一个变量里告诉你。

#### 2. 它是怎么工作的？

当你写下 `find_package(PCL REQUIRED)` 时，CMake 会做两件事：

1.  **寻找：** 在系统里找 PCL 的安装信息。
2.  **赋值：** 找到后，它会自动定义几个**标准变量**供你使用（注意全是**大写**）：
    -   `PCL_FOUND`: 找到了吗？（TRUE/FALSE）
    -   `PCL_INCLUDE_DIRS`: 头文件在哪里？（例如 `/usr/include/pcl-1.12`）
    -   `PCL_LIBRARIES`: 库文件在哪里？（例如 `/usr/lib/libpcl_common.so;...`）
    -   `PCL_DEFINITIONS`: 需要加什么预定义宏？（例如 `-DEIGEN_USE_NEW_STDVECTOR`）

#### 3. 怎么配合使用？

找到之后，你只需要把这些变量填进你的指令里：

```cmake
# 1. 召唤 PCL，如果没有安装 PCL，CMake 就在这里报错停止 (REQUIRED)
find_package(PCL REQUIRED)

# 2. 包含 PCL 的头文件路径 (自动填入)
include_directories(${PCL_INCLUDE_DIRS})

# 3. 添加 PCL 必需的宏定义 (自动填入)
add_definitions(${PCL_DEFINITIONS})

# 4. 生成可执行文件
add_executable(my_app main.cpp)

# 5. 链接 PCL 的库文件 (自动填入)
target_link_libraries(my_app ${PCL_LIBRARIES})
```

### 第二部分：CMake 常用指令指南 (Cheat Sheet)

我将这些指令分为**项目配置**、**寻找依赖**、**构建目标**、**辅助调试**四类，按使用频率排序。

#### 1. 项目基础配置 (必写)



| **指令**                     | **含义**               | **场景**                                              | **示例**                                                     |
| ---------------------------- | ---------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| **`cmake_minimum_required`** | 规定最低 CMake 版本    | **第一行必写**。防止低版本 CMake 无法识别你的新语法。 | `cmake_minimum_required(VERSION 3.10)`                       |
| **`project`**                | 定义项目名称和支持语言 | **第二行必写**。这会初始化很多内置变量。              | `project(PigBodyEstimation LANGUAGES CXX)`                   |
| **`set`**                    | 定义/修改变量          | 设置构建模式、C++ 标准等。                            | `set(CMAKE_CXX_STANDARD 17)` `set(CMAKE_BUILD_TYPE "Debug")` |

#### 2. 寻找与包含依赖 (核心)

| **指令**                         | **含义**           | **场景**                                                     | **示例**                                                     |
| :------------------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **`find_package`**               | 寻找外部库         | 用到 OpenCV, PCL, Eigen, ROS 时必用。                        | `find_package(OpenCV REQUIRED)`                              |
| **`include_directories`**        | 全局添加头文件路径 | **老式写法**。影响所有目标。简单项目可用，大项目不推荐。     | `include_directories(${PCL_INCLUDE_DIRS})`                   |
| **`target_include_directories`** | 局部添加头文件路径 | **现代写法 (推荐)**。只把头文件路径加给特定的程序，不污染全局。 | `target_include_directories(my_app PRIVATE ${PCL_INCLUDE_DIRS})` |

#### 3. 构建目标 (产出物)

| **指令**                    | **含义**       | **场景**                                             | **示例**                                           |
| --------------------------- | -------------- | ---------------------------------------------------- | -------------------------------------------------- |
| **`add_executable`**        | 生成可执行程序 | 你的 `main.cpp` 入口。                               | `add_executable(main_app main.cpp src/helper.cpp)` |
| **`add_library`**           | 生成库文件     | 把常用的算法封装成 `.so` 或 `.a`。                   | `add_library(my_algo SHARED src/algo.cpp)`         |
| **`target_link_libraries`** | 链接库文件     | **最重要**。告诉连接器，这个程序需要用到哪些库代码。 | `target_link_libraries(main_app ${OpenCV_LIBS})`   |

#### 4. 辅助与调试

| **指令**               | **含义**     | **场景**                                                     | **示例**                                          |
| ---------------------- | ------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| **`message`**          | 打印信息     | 调试 CMake 脚本用。类似于 C++ 的 `cout`。                    | `message(STATUS "PCL Path: ${PCL_INCLUDE_DIRS}")` |
| **`add_subdirectory`** | 添加子目录   | 当你的项目变大，有多个文件夹（如 `src`, `lib`, `test`）时，用它递归处理。 | `add_subdirectory(src)`                           |
| **`option`**           | 提供开关选项 | 允许用户在 `cmake ..` 时通过 `-D` 开关某些功能。             | `option(USE_CUDA "Enable CUDA support" OFF)`      |

### 第三部分：一个科研级 PCL 项目的“标准模板”

```cmake
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(PigBodyMeasure)

# 1. 设置 C++ 标准 (PCL 通常需要 C++14 或更高)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 设置默认构建模式为 Debug (方便你随时断点调试)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# 3. 寻找依赖库
# REQUIRED 表示找不到就报错停止，不要继续了
find_package(PCL 1.10 REQUIRED)
find_package(OpenCV 4.0 REQUIRED)

# 4. 包含头文件 (使用 include_directories 比较方便，虽然不如下面 target_ 严谨)
# PCL 和 OpenCV 找到后会自动填充这些变量
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 添加 PCL 的预定义宏 (这是 PCL 特有的要求)
add_definitions(${PCL_DEFINITIONS})

# 5. 定义可执行文件
# 假设你的源码都在 src 目录下
add_executable(pig_measure_app src/main.cpp src/process.cpp)

# 6. 链接库文件
# 将 PCL 和 OpenCV 的库链接到你的程序上
target_link_libraries(pig_measure_app
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)

# 7. (可选) 打印一条消息，确认配置成功
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PCL found at: ${PCL_INCLUDE_DIRS}")
```
