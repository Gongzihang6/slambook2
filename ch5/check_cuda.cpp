#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

int main() {
    int dev_count = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA Device Count: " << dev_count << std::endl;
    if (dev_count > 0) {
        cv::cuda::printShortCudaDeviceInfo(0);
    } else {
        std::cout << "No CUDA support or no GPU found." << std::endl;
    }
    return 0;
}
