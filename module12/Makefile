CXX       	:= g++
CXX_FLAGS 	:= -O3 -Wall -Wextra -std=c++17
LD_FLAGS 	:= -lOpenCL -lcrypto

TARGET 		:= assignment.exe
SRCS 		:= assignment.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $< -o $@ $(CXX_FLAGS) $(LD_FLAGS)

clean:
	rm -f $(TARGET)