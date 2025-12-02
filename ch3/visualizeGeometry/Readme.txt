1. 如何编译本程序：

* 使用 Pangolin：slambook/3rdpart/Pangolin 或从 GitHub 下载：https://github.com/stevenlovegrove/Pangolin

* 安装 Pangolin 的依赖（主要是 OpenGL）：
sudo apt-get install libglew-dev

* 编译并安装 Pangolin
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 
ldconfig

* 编译本程序：
mkdir build
cd build
cmake ..
make 

* 运行 build/visualizeGeometry

2. 如何使用本程序：

左侧面板展示了 T_w_c（相机到世界）的不同表示形式，包括旋转矩阵、平移向量、欧拉角（按 roll-pitch-yaw 顺序）以及四元数。
拖动鼠标左键可移动相机，右键可让相机绕盒子旋转，中键可旋转相机自身，同时按下左右键可滚动视图。
注意：在本程序中，原始 X 轴向右（红线），Y 轴向上（绿线），Z 轴向后（蓝线）。初始时，你（相机）站在 (3,3,3) 处，看向 (0,0,0)。

3. 可能遇到的问题：
* 我发现虚拟机中可能会出现 Pangolin 报错，该问题在其 issue 中已有解决方案：https://github.com/stevenlovegrove/Pangolin/issues/74。如遇到此问题，请按 paulinus 提到的方法注释掉两行代码，然后重新编译并安装 Pangolin。

如仍有使用问题，请联系：gaoxiang12@mails.tsinghua.edu.cn
