/*
 * 代码名称：Eigen 矩阵随机初始化与 Block 块操作示例
 * * 功能描述：
 * 1. 演示如何使用 C++11 的 <random> 库生成高质量的随机浮点数（替代老旧的 rand/srand）。
 * 2. 创建并填充一个 5x5 的 Eigen 动态矩阵。
 * 3. 演示 .block() 函数的用法，从大矩阵中提取子矩阵块。
 * 4. 演示对提取出的子矩阵进行赋值操作（如设为单位阵）。
 * * 核心实现：
 * - 随机数：使用 std::random_device 提供种子，std::default_random_engine 生成随机序列，std::uniform_real_distribution 控制范围。
 * - 矩阵块：使用 .block<Rows, Cols>(startRow, startCol) 方法提取。
 * * !!! 重要注意（坑点）!!!：
 * 代码中 `Matrix3d extractedBlock = ...` 这种写法执行的是【拷贝操作】。
 * 因此，修改 extractedBlock 不会影响原始的 bigMatrix。
 * 如果想要修改原矩阵，需要直接操作 block 或使用 Eigen::Ref。
 */
# include <iostream>
# include <Eigen/Core>

#include <random> // 使用 C++11 随机数库




using namespace std;
using namespace Eigen;

#define MATRIX_SIZE 5

int main(int argc, char **argv) {
    // // 设置随机种子， time（0）作为随机种子不安全
    // srand(static_cast<unsigned int>(time(0)));

    std::random_device rd;                          // 获取一个高质量的随机种子
    std::default_random_engine generator(rd());     // 初始化随机数生成器
    std::uniform_real_distribution<double> distribution(-1.0, 1.0); // 均匀分布 [-1, 1]


    cout.precision(3);

    // 初始化 Eigen 矩阵
    MatrixXd bigMatrix(MATRIX_SIZE, MATRIX_SIZE); // 生成一个 rows x cols 的矩阵

    // 3. 使用随机数填充矩阵
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            bigMatrix(i, j) = distribution(generator); // 填充每个元素
        }
    }

    cout << "The big matrix: \n" << bigMatrix << endl;

    // Eigen中用于提取子矩阵的.block()函数
    // 提取从第0行第0列开始的3x3子矩阵
    Matrix3d extractedBlock = bigMatrix.block<3,3>(0,0);

    cout << "The extracted matrix block: \n" << extractedBlock << endl;

    // 4. 对提取出的子矩阵进行赋值操作（如设为单位阵）
    // 注意：这里修改的是 extractedBlock 而不是 bigMatrix
    // 如果想要修改原矩阵，需要直接操作 block 或使用 Eigen::Ref
    extractedBlock.setIdentity();

    cout << "The assigned matrix block: \n" << extractedBlock << endl;

    // 使用引用 (Eigen::Ref) 这在函数传参时非常有用，可以避免拷贝。
    // 使用 Ref 创建一个“引用/视图”，指向原数据
    Ref<Matrix3d> blockRef = bigMatrix.block<3,3>(0,0);
    blockRef.setIdentity(); // 此时 bigMatrix 也会被修改！

    cout << "The modified big matrix: \n" << bigMatrix << endl;


    return 0;
}
