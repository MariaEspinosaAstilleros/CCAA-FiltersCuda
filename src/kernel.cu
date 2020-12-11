#include <stdio.h>

#include "../include/Filter.h"
#include "../include/kernel.h"

//kernel sobel filter
__global__ void kernel_filter(void){

    //Gradients of the sobel filter
    //int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    //int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("holaaa %d %d", x, y);


}

__host__ void Filter::sobel(){
    kernel_filter<<<1, 1>>>();
}