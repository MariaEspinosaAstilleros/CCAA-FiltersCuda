#include <opencv4/opencv2/imgcodecs.hpp>
#include <stdio.h>

__global__ void kernel_filters(unsigned char* src_img, unsigned char* dst_img, unsigned int width, unsigned int heigth);
__host__ void sobel_filter();
//__host__ void robert_filter();