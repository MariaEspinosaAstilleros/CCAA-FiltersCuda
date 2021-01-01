DIRSRC = ./src/
DIROBJ = obj/
DIREXE := exec/

CC = g++
CFLAGS = -I -c -pthread -std=c++11

NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_61,code=sm_61

INCLUDES = `pkg-config --cflags opencv4` -I/usr/local/cuda/include/ -I$(DIRSRC)
LIBS = `pkg-config --libs opencv4` -L/usr/local/cuda/libs/ -lcuda -lm 

SOURCE = main.cpp Filter.cpp
KERNEL = kernel.cu
OBJS = $(SOURCE:.cpp=.o) $(KERNEL:.cu=.o)
TARGET = filter

all: dirs $(TARGET)

dirs:
	mkdir -p $(DIROBJ) $(DIREXE)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(DIROBJ)main.o $(DIROBJ)Filter.o $(DIROBJ)kernel.o -o $(DIREXE)$@ $(INCLUDES) $(LIBS)

%.o: $(DIRSRC)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $(DIROBJ)$@ $(INCLUDES)

%.o: $(DIRSRC)/%.cpp
	$(CC) $(CFLAGS) -c $^ -o $(DIROBJ)$@ $(INCLUDES) $(LIBS)

test-sobel:
	./$(DIREXE)filter sobel

test-sharpen:
	./$(DIREXE)filter sharpen
		
clean:
	touch $(TARGET)
	rm -r $(DIREXE) $(DIROBJ) $(TARGET)
