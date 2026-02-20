all: assignment.cu
	nvcc assignment.cu aes_cpu.cc -o assignment.exe -Xcompiler "-fopenmp -lgomp -Wall -Wextra" -std=c++17
