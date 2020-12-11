#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <cuda.h>

#include "../include/kernel.h"

class Filter{
    private:
        cv::Mat src_image;
        std::string file_path;

        //CUDA hosts
        __host__ void sobel();
        //void robert_filter();

    public:
        Filter(std::string file_path);//constructor to apply filter in image
        //~Filter(); //destructor

        void sobel_filter();
        void write(std::string file_path);
};