#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <cuda.h> 

#include "../include/kernel.h"

class Filter{
    private:
        //CUDA host
        void applyFilter(cv::Mat* src_img, std::string type_filter);

    public:
        void optionPhoto(Filter filter, std::string type_filter);
        void optionCamera(Filter filter, std::string type_filter);
        void optionVideo(Filter filter, std::string type_filter); 

        void callFilter(cv::Mat src_img, std::string type_filter);

        cudaError_t testCuErr(cudaError_t dst_img);
};