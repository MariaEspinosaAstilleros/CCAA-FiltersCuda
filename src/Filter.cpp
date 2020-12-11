#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

#include "../include/Filter.h"
#include "../include/kernel.h"

/*Read to input image*/
Filter::Filter(std::string file_path): file_path(file_path){
    enum cv::ImreadModes mode; 
    mode = cv::IMREAD_GRAYSCALE;

    cv::Mat src_image = cv::imread(file_path, mode); 

    //Convert RGB to gray scale 
    //cv::cvtColor(src_image, src_image, cv::COLOR_RGB2GRAY);

    //TODO: cuando funcione todo tengo que comprobar que existe la imagen en el dir img


}

/*Apply sobel filter*/
void Filter::sobel_filter(){
    sobel();
}

/*Write final image*/
void Filter::write(std::string file_path){
    bool result = false; 

    //recorrer los datos para guardarlos en src_image

    try{
        result = cv::imwrite(file_path, src_image);
    }catch(const cv::Exception& err){
        fprintf(stderr, "Exception converting image: %s", err.what());
    }

   (result == true) ? "Saved image" : "ERROR. Can't save image"; 

}