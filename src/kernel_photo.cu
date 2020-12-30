#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include <../include/kernel_photo.h>
#include <../include/Filter.h>

#include <stdio.h>
#include <math.h>
#include <iostream>

#define BLOCK_SIZE 32
#define GRID_SIZE 128

/*Kernels*/
__global__ void kernelConvolutionSobel(unsigned char* src_img, unsigned char* dst_img, int width_img, int height_img){

    //Gradients of the sobel filter
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    int num_row = blockIdx.x * blockDim.x + threadIdx.x;
    int num_col = blockIdx.y * blockDim.y + threadIdx.y;

    int index = num_row * width_img + num_col;

    if(num_col < (width_img - 1) && num_row < (height_img - 1)){
        float grad_x= (src_img[index] * gx[0][0]) + (src_img[index+1] * gx[0][1]) + (src_img[index+2] * gx[0][2]) +
                      (src_img[index] * gx[1][0]) + (src_img[index+1] * gx[1][1]) + (src_img[index+2] * gx[1][2]) +
                      (src_img[index] * gx[2][0]) + (src_img[index+1] * gx[2][1]) + (src_img[index+2] * gx[2][2]);

        float grad_y= (src_img[index] * gy[0][0]) + (src_img[index+1] * gy[0][1]) + (src_img[index+2] * gy[0][2]) +
                      (src_img[index] * gy[1][0]) + (src_img[index+1] * gy[1][1]) + (src_img[index+2] * gy[1][2]) +
                      (src_img[index] * gy[2][0]) + (src_img[index+1] * gy[2][1]) + (src_img[index+2] * gy[2][2]);

        float gradient = sqrtf(grad_x * grad_x + grad_y * grad_y);

        if(gradient > 255)gradient = 255;
        if(gradient < 0)gradient = 0;

        dst_img[index] = gradient;
    }
}

__global__ void kernelConvolutionSharpen(unsigned char* src_img, unsigned char* dst_img, int width_img, int height_img){
    int num_row = blockIdx.x * blockDim.x + threadIdx.x;
    int num_col = blockIdx.y * blockDim.y + threadIdx.y;

    int index = num_row * width_img + num_col;
}

__host__ void Filter::applyFilter(cv::Mat *src_img, std::string type_filter){
    cudaFree(0);
    unsigned char *dev_src, *dev_sobel;
    int img_size = src_img->rows * src_img->cols * sizeof(unsigned char);
    cudaEvent_t start, end; 

    testCuErr(cudaEventCreate(&start)); testCuErr(cudaEventCreate(&end));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(GRID_SIZE, GRID_SIZE);

    testCuErr(cudaMalloc((void**)&dev_src, img_size));
    testCuErr(cudaMalloc((void**)&dev_sobel, img_size));

    //copy data to GPU
    testCuErr(cudaMemcpy(dev_src, src_img->data, img_size, cudaMemcpyHostToDevice));

    //start time
    testCuErr(cudaEventRecord(start));

    //call kernel
    if(type_filter.compare("sobel") == 0)
        kernelConvolutionSobel<<<numBlocks,threadsPerBlock>>>(dev_src, dev_sobel, src_img->cols, src_img->rows);

    if(type_filter.compare("sharpen") == 0)
        kernelConvolutionSharpen<<<numBlocks,threadsPerBlock>>>(dev_src, dev_sobel, src_img->cols, src_img->rows);

    testCuErr(cudaGetLastError());

    //end time 
    testCuErr(cudaEventRecord(end));
    testCuErr(cudaEventSynchronize(end));

    //diff time
    float diff = 0;
    testCuErr(cudaEventElapsedTime(&diff, start, end));
    std::cout << "Elapsed time: " << diff << " ms" << std::endl; 

    //copy data to CPU
    testCuErr(cudaMemcpy(src_img->data, dev_sobel, img_size, cudaMemcpyDeviceToHost));

    //free mem in device
    testCuErr(cudaFree(dev_src)); testCuErr(cudaFree(dev_sobel));
}