# `OpenCV`的`cv::parallel_for_`并行介绍

`OpenCV`的`cv::parallel_for_`是一个跨平台的并行计算框架，它能将一个大的循环任务自动拆解，利用多核CPU提升运行速度。

以第8讲中直接法估计相机位姿代码中：

```c++
cv::parallel_for_(
    cv::Range(0, px_ref.size()),
    std::bind(&JacobianAccumulator::accumulate_jacobian, 
    			&jaco_accu, 
    			std::placeholders::_1));
```

为例，这里面总共2个参数：

参数一：`cv::Range(0, px_ref.size())`——任务总空间

这个是告诉`OpenCV`，我们要处理的数据总共有多少个。这里的数据是从参考图中提取的特征像素点个数，我们的任务是计算每个特征像素点的灰度误差和，灰度误差对当前帧图像中像素坐标的雅可比导数，以及当前帧图像中像素坐标对变换矩阵（李代数）的雅可比导数。

参数二：`std::bind(…)`——任务执行体

这是一个回调函数（`Callback`），也就是告诉`OpenCV`，切分好区间后，每个线程要用这个数据干什么活。这里使用了C++11的`std::bind`来适配函数签名。以当前例子为例：

- `&JacobianAccumulator::accumulate_jacobian`是我们要并行调用的成员函数，它定义在ch8/direct_method.cpp中JacobianAccumulator类的`accumulate_jacobian`方法，作用就是计算每个像素点的海塞矩阵和雅可比矩阵和；
- `&jaco_accu`，因为是成员函数，所以我们必须知道是在哪个对象上调用（即`this`指针）。这里指定了是在`jaco_accu` 这个实例上执行；
- **`std::placeholders::_1`**：这是**占位符**。它表示：“这里有个参数我先不填，等`cv::parallel_for_` 真正调用的时候，它会把**切分好的子区间（Range）**填到这里。”

## `cv::parallel_for_` 的工作逻辑（图解）

假设 `px_ref.size()` 是 **1000**，你的电脑是 **4 核** CPU。`cv::parallel_for_` 的运行流程如下：

1. **任务拆分 (Splitting)**： `OpenCV` 内部的调度器会将 `[0, 1000)` 拆分成多个小块（Strips）。

    - 线程 1 拿到 Range: `[0, 250)`
    - 线程 2 拿到 Range: `[250, 500)`
    - 线程 3 拿到 Range: `[500, 750)`
    - 线程 4 拿到 Range: `[750, 1000)` *(注：具体的拆分策略取决于底层后端，可能是平均分，也可能是动态抢占)*

2. **并行执行 (Parallel Execution)**： `OpenCV` 启动 4 个工作线程，同时去调用你绑定的那个函数 `accumulate_jacobian`。

    - **Thread 1 调用**：`jaco_accu.accumulate_jacobian(Range(0, 250))`
    - **Thread 2 调用**：`jaco_accu.accumulate_jacobian(Range(250, 500))`
    - ...

3. **用户代码内的循环**： 这就是为什么你的 `accumulate_jacobian` 函数内部第一行必须是：

    ```c++
    for (size_t i = range.start; i < range.end; i++) { ... }
    ```

    因为它只负责处理分配给它的那一小段数据。

4. **阻塞等待 (Blocking)**： 主线程会卡在 `cv::parallel_for_` 这一行，直到这 4 个线程全部干完活，才会继续往下执行。

------

## 底层实现原理（Backend Agnostic）

`OpenCV` 自己并没有从零实现一套线程池，而是做了一个**虚拟层（Wrapper）**，它可以调用各种成熟的并行计算库。

**OpenCV 在编译时会检查系统环境，选择以下其中一种作为后端（Backend）：**

1. **Intel TBB (Threading Building Blocks)** —— **最常用/最高效**
    - 如果是 TBB，`OpenCV` 会把 `Range` 封装成 TBB 的 `blocked_range`，然后调用 `tbb::parallel_for`。
    - TBB 有极其高效的**任务窃取（Work Stealing）**机制。如果线程 A 处理完了自己的 250 个点，而线程 B 还在忙，线程 A 会偷偷把线程 B 剩下的任务抢一部分过来做。这保证了 CPU 不会闲置。
2. **OpenMP**
    - 经典的编译器级并行支持。`OpenCV` 会使用 `#pragma omp parallel for` 来实现。
3. **Apple GCD (Grand Central Dispatch)**
    - 在 macOS/iOS 上，调用系统的并发队列。
4. **C++11 std::thread / Pthreads**
    - 如果上面那些库都没装，`OpenCV` 会退化到一个简单的简易线程池实现，手动创建线程去分发 `Range`。

------

## 为什么 SLAM 代码里要这样写？（线程安全设计）

理解了原理，你再看 `JacobianAccumulator` 的设计，就会发现它是为了**线程安全**特意设计的：

1. **局部变量的使用**： 在 `accumulate_jacobian` 内部，定义了 `hessian`, `bias`, `cost_tmp`。

    ```c++
    // 这些是局部变量，分配在每个线程的栈（Stack）上
    // 线程之间互不干扰，不需要加锁！
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    ```

2. **最后的加锁合并**： 只有当循环结束，要将**局部结果**合并到**全局对象**（`jaco_accu` 的成员变量）时，才需要加锁。

    ```c++
    if (cnt_good) {
        // 这一步必须加锁，因为 4 个线程可能同时运行到这里
        // 如果不加锁，同时写 H += hessian，数据就乱了
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
    }
    ```

## 总结

- **`cv::parallel_for_`**：是一个**分发器**，负责把大循环切成小块，扔进线程池。
- **`cv::Range`**：是**切块的边界**，告诉子线程“你只负责处理这一段”。
- **`std::bind`**：是**胶水**，把你的成员函数包装成 `OpenCV` 能调用的标准格式。
- **性能关键**：利用多核并行计算 Hessian 矩阵（计算密集型任务），通常能带来 4-8 倍的加速（取决于核心数）。

