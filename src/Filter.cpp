#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include <../include/kernel_photo.h>
#include "../include/Filter.h"
#include <../include/colors.h>

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define BLOCK_SIZE 32
#define GRID_SIZE 128

cudaError_t Filter::testCuErr(cudaError_t dst_img){
    if (dst_img != cudaSuccess) {
        printf("CUDA Runtime Error: %s\n", 
            cudaGetErrorString(dst_img));
        assert(dst_img == cudaSuccess);
    }
    return dst_img;
}

void Filter::optionPhoto(Filter filter){
    std::string input_img_path;
    std::cout << "Select a photo to apply the sobel filter:" << std::endl;
    std::cin >> input_img_path;
    cv::Mat src_img = cv::imread(input_img_path, cv::IMREAD_GRAYSCALE);
   
    if(!src_img.data){
        std::cerr << "ERROR. No image data." << std::endl;
        std::cout << "Enter path that contains the image: " << YELLOW << "img/<name_image>" << RESET << std::endl;
        exit(-1);
    }else{
        filter.sobelFilter(src_img); // apply sobel filter to the photo

        cv::resize(src_img, src_img, cv::Size(1366,768));
        cv::imshow("CUDA Sobel", src_img);
        cv::waitKey(0);
    }
}

void Filter::optionCamera(){
    cv::VideoCapture camera(0);

    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        exit(-1);
    }

    while (true){ 
        cv::Mat cam_frame;
        
        //camera.read(cam_frame);
        camera >> cam_frame;
        cv::imshow("CUDA Sobel WebCam", cam_frame);
        if (cv::waitKey(10) >= 0)
        break;
    }
}

void Filter::optionVideo(){

}

void Filter::sobelFilter(cv::Mat src_img){
    sobel(&src_img);
}