/*Functions definitions*/
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

cudaError_t testCuErr(cudaError_t result);
int iDivUp(int a, int b);
void sobelFilterPhoto(cv::Mat *src_image);

