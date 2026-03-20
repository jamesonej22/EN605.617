NVCC        := nvcc
CXX_FLAGS   := -Wall -Wextra
NVCC_FLAGS  := -O3 -std=c++17 -Xcompiler "$(CXX_FLAGS)" \
               -lcurand -lcublas -lcusolver --extended-lambda

TARGET      := assignment.exe
SRCS        := assignment.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $< -o $@ $(NVCC_FLAGS)

clean:
	rm -f $(TARGET)