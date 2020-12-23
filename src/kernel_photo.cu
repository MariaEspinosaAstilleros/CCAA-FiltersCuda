#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include <../include/kernel_photo.h>
#include <../include/colors.h>

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define BLOCK_SIZE 32
#define GRID_SIZE 128

/*Kernel*/
__global__ void kernel_convolution_sobel(char* src_img, char* dst_img, int width_img, int height_img){

    //Gradients of the sobel filter
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    int num_col = blockIdx.x * blockDim.x + threadIdx.x;
    int num_row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = (num_row * width_img) + num_col;

    if(num_col < (width_img - 1) && num_row < (height_img - 1)) {
        int grad_x = (src_img[index] * gx[0][0]) + (src_img[index] * gx[0][1]) + (src_img[index] * gx[0][2]) +
                     (src_img[index + 1] * gx[1][0]) + (src_img[index + 1] * gx[1][1]) + (src_img[index + 1] * gx[1][2]) +
                     (src_img[index + 2] * gx[2][0]) + (src_img[index + 2] * gx[2][1]) + (src_img[index + 2] * gx[2][2]);

        int grad_y = (src_img[index] * gy[0][0]) + (src_img[index] * gy[0][1]) + (src_img[index] * gy[0][2]) +
                     (src_img[index + 1] * gy[1][0]) + (src_img[index + 1] * gy[1][1]) + (src_img[index + 1] * gy[1][2]) +
                     (src_img[index + 2] * gy[2][0]) + (src_img[index + 2] * gy[2][1]) + (src_img[index + 2] * gy[2][2]);

        dst_img[index] = (char) sqrtf(grad_x * grad_x + grad_y * grad_y);
    }
}

/*Main*/
int main(int argc, char **argv){ 
    std::string input_img_path = argv[1];
    cv::Mat src_img = cv::imread(input_img_path, cv::IMREAD_GRAYSCALE);

    sobelFilterPhoto(&src_img); // apply sobel filter to the photo

    cv::resize(src_img, src_img, cv::Size(1366,768));
    cv::imshow("CUDA Sobel", src_img);
    cv::waitKey(0);

    return 0;
}

/* Auxiliar functions*/
cudaError_t testCuErr(cudaError_t dst_img){
    if (dst_img != cudaSuccess) {
        printf("CUDA Runtime Error: %s\n", 
            cudaGetErrorString(dst_img));
        assert(dst_img == cudaSuccess);
    }
    return dst_img;
}

void sobelFilterPhoto(cv::Mat *src_img){

    //check input image
    if(!src_img->data){
        std::cout << "No image data." << std::endl;
        std::cout << "Enter path that contains the image: " << YELLOW << "img/<name_image>" << RESET << std::endl;
        exit(-1);
    }

    cudaFree(0);
    char *dev_src, *dev_sobel;
    int img_size = src_img->rows * src_img->cols * sizeof(char);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(GRID_SIZE, GRID_SIZE);

    testCuErr(cudaMalloc((void**)&dev_src, img_size));
    testCuErr(cudaMalloc((void**)&dev_sobel, img_size));

    //copy data to GPU
    testCuErr(cudaMemcpy(dev_src, src_img->data, img_size, cudaMemcpyHostToDevice));

    //start time
    auto start = std::chrono::high_resolution_clock::now();

    //call kernel
    kernel_convolution_sobel<<<numBlocks,threadsPerBlock>>>(dev_src, dev_sobel, src_img->step1(), src_img->rows);
    testCuErr(cudaGetLastError()); testCuErr(cudaDeviceSynchronize());

    //end time 
    auto end = std::chrono::high_resolution_clock::now(); 

    //diff time
    std::chrono::duration<double> diff = end - start;
    std::cout << "Elapsed time: " << diff.count() << " seg" << std::endl;

    //copy data to CPU
    testCuErr(cudaMemcpy(src_img->data, dev_sobel, img_size, cudaMemcpyDeviceToHost));

    //free mem in device
    testCuErr(cudaFree(dev_src)); testCuErr(cudaFree(dev_sobel));
}
