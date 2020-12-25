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
    std::cout << "Select an option: photo " << MAGENTA << "(1) " << RESET", camera "  << MAGENTA << "(2) " << RESET "or video " << MAGENTA << "(3) " << RESET << std::endl;
    std::cin >> option;

    Filter filter(); //create object filter

    switch(option){
         
        case 1: 
            filter.optionPhoto(filter);
            break;
        case 2: 
            filter.optionCamera();
            break;
        case 3: 
            filter.optionVideo();
            break;
    }
    return 0;

}

