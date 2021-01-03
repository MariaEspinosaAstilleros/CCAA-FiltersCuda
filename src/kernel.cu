#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include <../include/kernel.h>
#include <../include/Filter.h>
#include <../include/colors.h>

#include <stdio.h>
#include <math.h>
#include <iostream>

#define BLOCK_SIZE 32
#define GRID_SIZE 128
#define KERNEL_SIZE 3

/*Kernels*/
__global__ void kernelConvolutionSobel(unsigned char* src_img, unsigned char* dst_img, int width_img, int height_img){

    int sobel_x[KERNEL_SIZE][KERNEL_SIZE] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[KERNEL_SIZE][KERNEL_SIZE] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    int num_row = blockIdx.x * blockDim.x + threadIdx.x;
    int num_col = blockIdx.y * blockDim.y + threadIdx.y;

    int index = num_row * width_img + num_col;

    if(num_col < (width_img - 1) && num_row < (height_img - 1)){
        float grad_x= (src_img[index] * sobel_x[0][0]) + (src_img[index+1] * sobel_x[0][1]) + (src_img[index+2] * sobel_x[0][2]) +
                      (src_img[index] * sobel_x[1][0]) + (src_img[index+1] * sobel_x[1][1]) + (src_img[index+2] * sobel_x[1][2]) +
                      (src_img[index] * sobel_x[2][0]) + (src_img[index+1] * sobel_x[2][1]) + (src_img[index+2] * sobel_x[2][2]);

        float grad_y= (src_img[index] * sobel_y[0][0]) + (src_img[index+1] * sobel_y[0][1]) + (src_img[index+2] * sobel_y[0][2]) +
                      (src_img[index] * sobel_y[1][0]) + (src_img[index+1] * sobel_y[1][1]) + (src_img[index+2] * sobel_y[1][2]) +
                      (src_img[index] * sobel_y[2][0]) + (src_img[index+1] * sobel_y[2][1]) + (src_img[index+2] * sobel_y[2][2]);

        float gradient = sqrtf(grad_x * grad_x + grad_y * grad_y);

        if(gradient > 255) gradient = 255;
        if(gradient < 0) gradient = 0;

        __syncthreads();

        dst_img[index] = gradient;
    }
}

__global__ void kernelConvolutionSharpen(unsigned char* src_img, unsigned char* dst_img, int width_img, int height_img){

    int sharpen[KERNEL_SIZE][KERNEL_SIZE] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};

    int num_row = blockIdx.x * blockDim.x + threadIdx.x;
    int num_col = blockIdx.y * blockDim.y + threadIdx.y;

    int index = num_row * width_img + num_col;

    if(num_col < (width_img - 1) && num_row < (height_img - 1)){
        float sum = (src_img[index] * sharpen[0][0]) + (src_img[index+1] * sharpen[0][1]) + (src_img[index+2] * sharpen[0][2]) +
                    (src_img[index] * sharpen[1][0]) + (src_img[index+1] * sharpen[1][1]) + (src_img[index+2] * sharpen[1][2]) +
                    (src_img[index] * sharpen[2][0]) + (src_img[index+1] * sharpen[2][1]) + (src_img[index+2] * sharpen[2][2]);

        if(sum > 255) sum = 255;
        if(sum < 0)sum = 0;

        __syncthreads();

        dst_img[index] = sum;
    }
}

cudaError_t Filter::testCuErr(cudaError_t dst_img){
    if (dst_img != cudaSuccess) {
        printf("CUDA Runtime Error: %s\n", 
            cudaGetErrorString(dst_img));
        assert(dst_img == cudaSuccess);
    }
    return dst_img;
}

__host__ void Filter::applyFilter(cv::Mat *src_img, std::string type_filter){
    cudaFree(0);
    unsigned char  *dev_src, *dev_sobel;
    int img_size = src_img->rows * src_img->cols * sizeof(unsigned char);
    cudaEvent_t start, end; 

    testCuErr(cudaEventCreate(&start)); testCuErr(cudaEventCreate(&end));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(GRID_SIZE, GRID_SIZE);

    testCuErr(cudaMalloc((void**)&dev_src, img_size));
    testCuErr(cudaMalloc((void**)&dev_sobel, img_size));

    testCuErr(cudaMemcpy(dev_src, src_img->data, img_size, cudaMemcpyHostToDevice));

    testCuErr(cudaEventRecord(start));

    if(type_filter.compare("sobel") == 0)
        kernelConvolutionSobel<<<numBlocks,threadsPerBlock>>>(dev_src, dev_sobel, src_img->cols, src_img->rows);

    if(type_filter.compare("sharpen") == 0)
        kernelConvolutionSharpen<<<numBlocks,threadsPerBlock>>>(dev_src, dev_sobel, src_img->cols, src_img->rows);

    testCuErr(cudaGetLastError());

    testCuErr(cudaEventRecord(end));
    testCuErr(cudaEventSynchronize(end));

    float milliseconds = 0;
    testCuErr(cudaEventElapsedTime(&milliseconds, start, end));
    std::cout << CYAN << "Elapsed time: "  << RESET << milliseconds << " ms" << std::endl; 

    testCuErr(cudaMemcpy(src_img->data, dev_sobel, img_size, cudaMemcpyDeviceToHost));

    testCuErr(cudaFree(dev_src)); testCuErr(cudaFree(dev_sobel));
}