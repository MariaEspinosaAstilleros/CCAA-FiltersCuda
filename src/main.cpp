#include "../include/Filter.h"
#include <../include/colors.h>

#include <stdio.h>
#include <iostream>

int main(int argc, char **argv){
    std::string type_filter = argv[1];
    Filter      filter;
    int         option;

    if((type_filter.compare("sobel") && type_filter.compare("sharpen")) != 0){ //check type filter
        std::cout << "Sorry. You must select " << YELLOW << "sobel " << RESET << "filter or " << YELLOW << "sharpen " << RESET << "filter." << std::endl;
        exit(-1);
    }

    std::cout << "Select an option: " << std::endl;
    std::cout << MAGENTA << "1) " << RESET << "Photo" << std::endl;
    std::cout << MAGENTA << "2) " << RESET << "Webcam" << std::endl;
    std::cout << MAGENTA << "3) " << RESET << "Video" << std::endl;
    std::cin >> option;

    switch(option){
        case 1: 
            filter.optionPhoto(filter, type_filter);
            break;
        case 2: 
            filter.optionCamera(filter, type_filter);
            break;
        case 3: 
            filter.optionVideo(filter, type_filter);
            break;
    }
    return 0;
}

