/*Functions definitions*/
#include <iostream>
#include "cuda_runtime.h"

__global__ void kernelConvolutionSobel(unsigned char* src_img, unsigned char* dst_img, int width_img, int height_img);
__global__ void kernelConvolutionSharpen(unsigned char* src_img, unsigned char* dst_img, int width_img, int height_img);



