#include <stdio.h>
#include "cuda_runtime.h"

__global__ void kernel_filters(void);
__host__ void sobel(); 