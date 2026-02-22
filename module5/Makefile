all: assignment.cu
	nvcc assignment.cu aes_cpu.cc -o assignment.exe -Xcompiler "-fopenmp -lgomp -Wall -Wextra -O3" -std=c++17
