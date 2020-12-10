#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

//kernel sobel filter
__global__ void kernel_filters(unsigned char* src_img, unsigned char* dst_img, unsigned int width, unsigned int heigth){

    /*Gradients of the sobel filter*/
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    int x = blockId.x * blockDim.x + threadId.x; 
    int y = blockId.y * blockDim.y + threadd.y;

    printf("Hola jeje");

}

__host__ void Filter::sobel_filter(){

}

/*__host__ void Filter::robert_filter(){

}*/