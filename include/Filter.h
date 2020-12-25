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
        //void robert_filter();

    public:
        Filter();//constructor
        //~Filter(); //destructor

        void optionPhoto(Filter filter);
        void optionCamera();
        void optionVideo(); 
        void sobelFilter(cv::Mat);
        cudaError_t testCuErr(cudaError_t dst_img);
};