#include <iostream>

#include <opencv2/imgcodecs.hpp>

//kernel sobel filter
__global__ void kernel_sobel(unsigned char* src_img, unsigned char* dst_img, unsigned int width, unsigned int heigth){

    /*Gradients of the sobel filter*/
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}; 

}