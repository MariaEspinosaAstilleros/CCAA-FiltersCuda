#include <opencv4/opencv2/imgproc.hpp> //to cv::cvtColor(src, dst, mode_color, cv::COLOR_RGB2GRAY)
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "../include/Filter.h"

int main(int argc, char** argv){

    std::string file_path;
    std::cout << "Enter the path of the image:" << std::endl; 
    std::cin >> file_path; 

    Filter image(file_path);

    return 0;
}