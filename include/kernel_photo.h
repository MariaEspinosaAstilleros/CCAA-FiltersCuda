/*Functions definitions*/
#include <iostream>
#include "cuda_runtime.h"

__global__ void kernelConvolutionSobel(unsigned char* src_img, unsigned char* dst_img, int width_img, int height_img);
__host__ void sobelFilter();
__host__ void otherFilter();



