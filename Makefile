PREFIX =./src/

CC = g++
CFLAGS = -I -c -pthread -std=c++11

NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_61,code=sm_61

# Change here with the correct path for the CUDA and OpenCV installation
# (pkg-config will try to find the opencv4 installation)
INCLUDES = `pkg-config --cflags opencv4` -I/usr/local/cuda/include/ -I$(PREFIX)
LIBS = `pkg-config --libs opencv4` -L/usr/local/cuda/libs/ -lcuda -lm 

SOURCE = main.cpp Filter.cpp
KERNEL = kernel.cu
OBJS = $(SOURCE:.cpp=.o) $(KERNEL:.cu=.o)
TARGET = filter_sobel

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

%.o: $(PREFIX)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@ $(INCLUDES)

%.o: $(PREFIX)/%.cpp
	$(CC) $(CFLAGS) -c $^ -o $@ $(INCLUDES) $(LIBS)
		
clean:
	touch $(TARGET)
	rm *.o $(TARGET)