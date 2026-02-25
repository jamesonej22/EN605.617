NVCC        := nvcc
CXX_FLAGS   := -Wall -Wextra
NVCC_FLAGS  := -Xcompiler "$(CXX_FLAGS)" -O3 -std=c++17

TARGET      := assignment.exe
SRCS        := assignment.cu aes.cu
OBJS        := $(SRCS:.cu=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

%.o: %.cu
	$(NVCC) --device-c $< -o $@ $(NVCC_FLAGS)

clean:
	rm -f $(OBJS) $(TARGET) encrypted_*.ppm