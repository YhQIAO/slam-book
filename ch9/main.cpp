#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "Hello, World!" << std::endl;
    cv::Mat a(500,500,CV_8UC1);
    for(int i =0 ;i < 500;i++) {
        for(int j = 0;j < 500;j++) {
            if(i*j > 10000) {
                a.at<uchar>(i,j) = 0;
            } else {
                a.at<uchar>(i,j) = 255;
            }
        }
    }
    cv::imshow("a", a);
    cv::waitKey();
    return 0;
}
