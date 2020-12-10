#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

class Filter{
    private:
        cv::Mat src_image;
        std::string file_path;

        //CUDA hosts
        void sobel_filter();
        void robert_filter();

    public:
        Filter(std::string file_path);//constructor to apply filter in image
        //~Filter(); //destructor
};