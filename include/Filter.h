#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <cuda.h> 

#include "../include/kernel_photo.h"

class Filter{
    private:
        cv::Mat src_image;
        std::string file_path;

        //CUDA hosts
        void sobel(cv::Mat*);
        void other();

    public:
        void optionPhoto(Filter filter);
        void optionCamera(Filter filter);
        void optionVideo(Filter filter); 
        void sobelFilter(cv::Mat);
        cudaError_t testCuErr(cudaError_t dst_img);
};