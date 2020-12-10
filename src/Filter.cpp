#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

#include "../include/kernel.h"
#include "../include/Filter.h"

/*Read to input image*/
Filter::Filter(std::string file_path): file_path(file_path){
    enum cv::ImreadModes mode; 
    mode = cv::IMREAD_GRAYSCALE;

    cv::Mat src_image = cv::imread(file_path, mode);

    std::cout << "hola" << std::endl; 

    //Convert RGB to gray scale 
    //cv::cvtColor(src_image, src_image, cv::COLOR_RGB2GRAY);
}