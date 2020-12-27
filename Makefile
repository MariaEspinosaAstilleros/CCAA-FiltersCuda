DIRSRC = ./src/
DIROBJ = obj/

CC = g++
CFLAGS = -I -c -pthread -std=c++11

NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_61,code=sm_61

INCLUDES = `pkg-config --cflags opencv4` -I/usr/local/cuda/include/ -I$(DIRSRC)
LIBS = `pkg-config --libs opencv4` -L/usr/local/cuda/libs/ -lcuda -lm 

SOURCE = main.cpp Filter.cpp
KERNEL = kernel_photo.cu
OBJS = $(SOURCE:.cpp=.o) $(KERNEL:.cu=.o)
TARGET = filter_sobel

all: dirs $(TARGET)

dirs:
	mkdir obj

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(DIROBJ)main.o $(DIROBJ)Filter.o $(DIROBJ)kernel_photo.o -o $@ $(INCLUDES) $(LIBS)

%.o: $(DIRSRC)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $(DIROBJ)$@ $(INCLUDES)

%.o: $(DIRSRC)/%.cpp
	$(CC) $(CFLAGS) -c $^ -o $(DIROBJ)$@ $(INCLUDES) $(LIBS)

run:
	./filter_sobel 
		
clean:
	touch $(TARGET)
	rm -r $(DIROBJ) $(TARGET)
