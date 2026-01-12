/**
 * 在 SLAM 系统开发中，Config 类解决了三个核心痛点：

1. 参数与代码解耦 (Decoupling)
没 Config 前：你可能会把参数硬编码在代码里，比如 double fx = 718.856;。每次换个相机或者调个阈值，你都要修改 .cpp 源码重新编译，效率极低。

有 Config 后：参数写在 default.yaml 文本文件里。调参只需要改文本文件，不需要重新编译代码。这对于现场调试（Tuning）至关重要。

2. 避免参数传递的“噩梦”
问题：假设你的后端优化模块 Backend 需要一个参数 chi2_th。

main() 读了参数 -> 传给 VisualOdometry -> 传给 Backend -> 传给 Optimize 函数。

为了一个参数，你需要在四五层函数的参数列表里加上它，代码耦合度极高，非常丑陋。

解决方案 (Global Access)：由于 Config 是单例的，且方法是 static 的，这意味着整个项目的任何角落（无论是前端、后端还是地图模块），只要 #include "myslam/config.h"，就可以直接调用 Config::Get(...) 拿到参数。这极大简化了接口设计。

3. 统一管理
所有参数集中在一个 YAML 文件中，方便查看和管理。如果参数散落在各个 .cpp 文件的宏定义里，过两个月你自己都找不到哪里改参数。
 */

#pragma once
#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"      // 因为需要用到 cv::FileStorage（OpenCV 的文件读取类）和智能指针

namespace myslam {

/**
 * 配置类，使用SetParameterFile确定配置文件
 * 然后用Get得到对应值
 * 单例模式
 */
class Config {
private:
    // 1. 静态成员变量：唯一的实例指针
    // 这是一个静态指针，它是全村的希望，所有人都通过访问这个静态变量来获取唯一的 Config 实例
    static std::shared_ptr<Config> config_;

    // 2. 核心成员：OpenCV的文件存储对象
    cv::FileStorage file_;

    // 3. 私有构造函数：禁止外部随便 new Config()
    /**
     * 设计模式：单例模式（Singleton Pattern）
     * 核心思想：保证一个类仅有一个实例，并提供一个访问它的全局访问点
     * 为什么 private 构造函数？ 
     *      如果构造函数是 public 的，也就是公开的，那么谁都可以在代码的任何地方写 new Config()，
     *  这就会导致内存里有多个 Config 对象，不仅浪费内存，还可能导致不同模块读取的配置不一致。把它设为 private，就只有 Config 类自己能创建自己。
     */
    Config() {}  // private constructor makes a singleton
public:
    // 析构函数：关闭文件   ，资源获取即初始化 (Resource Acquisition Is Initialization)
    // 当程序结束，config_ 智能指针引用计数归零，自动调用析构函数。析构函数里会执行 file_.release() 关闭文件句柄，防止资源泄露。
    ~Config();  // close the file when deconstructing

    // set a new config file，// 设置配置文件（初始化单例）
    /**
     * 它负责两件事：
     *      检查 config_ 是否为空，如果为空就创建一个新的 Config 实例（懒汉式加载）。
     *      调用 file_.open(filename, ...) 打开 YAML 配置文件。
     */
    static bool SetParameterFile(const std::string &filename);

    // access the parameter values
    /**
     * 配置文件里的值类型千奇百怪，有 int (迭代次数), double (相机焦距), string (数据集路径), float 等。
     * 如果我们为每种类型都写一个函数 GetInt, GetDouble, GetString，代码会非常冗余
     * 解决方案：使用模板 typename T
     */
    template <typename T>
    static T Get(const std::string &key) {
        return T(Config::config_->file_[key]);
    }
};
}  // namespace myslam

#endif  // MYSLAM_CONFIG_H
