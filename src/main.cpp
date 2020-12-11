#include <opencv4/opencv2/imgproc.hpp> //to cv::cvtColor(src, dst, mode_color, cv::COLOR_RGB2GRAY)
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "../include/Filter.h"

int main(int argc, char** argv){

    if(argc != 2){
        std::cout << "Invalid number of command line arguments..." << std::endl;
        std::cout << "Usage: " << argv[0] << " [image.png]"<< std::endl;
        return EXIT_FAILURE;
    } 

    Filter image(argv[1]);
    image.sobel_filter();
    image.write("img/outSobel.jpg");

    return EXIT_SUCCESS;
}