#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include "../include/Filter.h"
#include <../include/colors.h>

#include <stdio.h>
#include <iostream>

int main(int argc, char **argv){
    int option;
    std::cout << "Select an option: " << std::endl;
    std::cout << MAGENTA << "1) " << RESET << "Photo" << std::endl;
    std::cout << MAGENTA << "2) " << RESET << "Webcam" << std::endl;
    std::cout << MAGENTA << "3) " << RESET << "Video" << std::endl;
    std::cin >> option;

    Filter filter;

    switch(option){
         
        case 1: 
            filter.optionPhoto(filter);
            break;
        case 2: 
            filter.optionCamera(filter);
            break;
        case 3: 
            filter.optionVideo(filter);
            break;
    }
    return 0;
}

