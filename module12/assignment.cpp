#define CL_TARGET_OPENCL_VERSION 220

#include <openssl/evp.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define NUM_SIZES 4
#define NUM_RUNS 5
#define WARMUP_RUNS 1
#define CHACHA_KEY_SIZE 32
#define CHACHA_NONCE_SIZE 12

const uint32_t INPUT_SIZES[NUM_SIZES] = {1 << 20, 1 << 24, 1 << 28, 1 << 30};

void chacha20_openssl(const uint8_t *key, const uint8_t *nonce, uint32_t counter,
                      const uint8_t *input, uint8_t *output, size_t len) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();

    uint8_t iv[16];
    std::memcpy(iv + 4, nonce, 12);
    iv[0] = counter & 0xFF;
    iv[1] = (counter >> 8) & 0xFF;
    iv[2] = (counter >> 16) & 0xFF;
    iv[3] = (counter >> 24) & 0xFF;

    EVP_EncryptInit_ex(ctx, EVP_chacha20(), NULL, key, iv);

    int out_len;
    EVP_EncryptUpdate(ctx, output, &out_len, input, len);

    EVP_CIPHER_CTX_free(ctx);
}

void random_bytes(uint8_t *data, size_t size, std::mt19937_64 &rng) {
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);

    size_t i = 0;
    while (i + 4 <= size) {
        uint32_t r = dist(rng);
        data[i++] = r & 0xFF;
        data[i++] = (r >> 8) & 0xFF;
        data[i++] = (r >> 16) & 0xFF;
        data[i++] = (r >> 24) & 0xFF;
    }

    while (i < size) {
        data[i++] = dist(rng) & 0xFF;
    }
}

cl_context create_context() {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM,
                                                 (cl_context_properties)firstPlatformId, 0};
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context =
            clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS) {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

cl_command_queue create_command_queue(cl_context context, cl_device_id *device) {
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue command_queue = NULL;
    size_t deviceBufferSize = -1;

    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed call to clGetContextInfo(..., GL_CONTEXT_DEVICES, ...)";
        return NULL;
    }

    if (deviceBufferSize <= 0) {
        std::cerr << "No devices available.";
        return NULL;
    }

    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS) {
        delete[] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    command_queue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);
    if (command_queue == NULL) {
        delete[] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete[] devices;
    return command_queue;
}

cl_program create_program(cl_context context, cl_device_id device, const char *fileName) {
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);
    if (program == NULL) {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog,
                              NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

int initialize(cl_context &context, cl_command_queue &command_queue, cl_program &program,
               cl_kernel &kernel) {
    cl_device_id device = 0;

    context = create_context();
    if (context == NULL) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    command_queue = create_command_queue(context, &device);
    if (command_queue == NULL) {
        std::cerr << "Failed to create OpenCL command queue." << std::endl;
        return 1;
    }

    program = create_program(context, device, "chacha20.cl");
    if (program == NULL) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        return 1;
    }

    kernel = clCreateKernel(program, "chacha_kernel", NULL);
    if (kernel == NULL) {
        std::cerr << "Failed to create OpenCL kernel." << std::endl;
        return 1;
    }

    return 0;
}

int create_buffers(cl_context context, uint32_t input_size, cl_mem &input, cl_mem &output,
                   cl_mem &key_buf, cl_mem &nonce_buf, const uint8_t *plaintext,
                   const uint8_t *key_host, const uint8_t *nonce_host) {
    input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size,
                           (void *)plaintext, NULL);

    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input_size, NULL, NULL);

    key_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, (void *)key_host,
                             NULL);

    nonce_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 12,
                               (void *)nonce_host, NULL);

    if (!input || !output || !key_buf || !nonce_buf) {
        std::cerr << "Error creating buffers\n";
        return 1;
    }

    return 0;
}

int run_kernel(cl_command_queue command_queue, cl_kernel kernel, cl_mem input, cl_mem output,
               cl_mem key_buf, cl_mem nonce_buf, uint32_t input_size, uint32_t counter,
               uint8_t *ciphertext) {
    cl_int errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &key_buf);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &nonce_buf);
    errNum |= clSetKernelArg(kernel, 4, sizeof(uint32_t), &counter);
    errNum |= clSetKernelArg(kernel, 5, sizeof(uint32_t), &input_size);

    if (errNum != CL_SUCCESS) {
        std::cerr << "Error setting kernel arguments.\n";
        return 1;
    }

    size_t num_blocks = (input_size + 63) / 64;
    size_t globalWorkSize[1] = {num_blocks};

    errNum =
        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error launching kernel.\n";
        return 1;
    }

    clFinish(command_queue);

    errNum = clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, input_size, ciphertext, 0, NULL,
                                 NULL);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error reading buffer.\n";
        return 1;
    }

    return 0;
}

void cleanup(cl_context context, cl_command_queue command_queue, cl_program program,
             cl_kernel kernel, cl_mem input, cl_mem output, cl_mem key, cl_mem nonce) {
    if (input != 0) clReleaseMemObject(input);
    if (output != 0) clReleaseMemObject(output);
    if (key != 0) clReleaseMemObject(key);
    if (nonce != 0) clReleaseMemObject(nonce);
    if (command_queue != 0) clReleaseCommandQueue(command_queue);
    if (kernel != 0) clReleaseKernel(kernel);
    if (program != 0) clReleaseProgram(program);
    if (context != 0) clReleaseContext(context);
}

void print_header(unsigned long long seed) {
    std::cout
        << "\n+-----------------------------------------------------------------------------+\n"
        << "| EN605.617 Module 12 Assignment                                 Eric Jameson |\n"
        << "+-----------------------------------------------------------------------------+\n"
        << "| Performing OpenCL-based ChaCha20 encryption vs. OpenSSL reference           |\n"
        << "| implementation on random byte arrays of sizes:                              |\n"
        << "| ";

    for (int i = 0; i < NUM_SIZES; i++) {
        std::cout << INPUT_SIZES[i];
        if (i != NUM_SIZES - 1) {
            std::cout << ", ";
        }
    }
    std::cout
        << std::setw(38) << " |\n"
        << "|                                                                             |\n"
        << "| Random seed: " << seed << std::setw(65 - std::to_string(seed).length()) << " |\n"
        << "+-----------------------------------------------------------------------------+\n";
}

void print_timing(uint32_t input_size, double total_time_host, double total_time_gpu, bool match) {
    double avg_time_gpu = total_time_gpu / NUM_RUNS;
    double gb = input_size / (1024.0 * 1024.0 * 1024.0);
    double seconds = avg_time_gpu / 1000.0;
    double throughput_gpu = gb / seconds;

    double avg_time_host = total_time_host / NUM_RUNS;
    gb = input_size / (1024.0 * 1024.0 * 1024.0);
    seconds = avg_time_host / 1000.0;
    double throughput_host = gb / seconds;
    std::string match_string = match ? "Outputs match!" : "Mismatch in output";
    int total_width = 77;
    int value_width = 14;

    std::cout << std::fixed << std::setprecision(5);
    std::cout
        << "| INPUT SIZE: " << input_size
        << std::setw(total_width - std::to_string(input_size).length() - match_string.length())
        << match_string << " |\n"
        << "+-----------------------------------------------------------------------------+\n"
        // OpenCL
        << "|   OpenCL    |                     Avg Time:        " << std::setw(value_width + 3)
        << avg_time_gpu << " ms     |\n"
        << "|             |                     Throughput:      " << std::setw(value_width + 3)
        << throughput_gpu << " GB/s   |\n"
        << "+-------------+                                                               |\n"
        // OpenSSL
        << "|   OpenSSL   |                     Avg Time:        " << std::setw(value_width + 3)
        << avg_time_host << " ms     |\n"
        << "|             |                     Throughput:      " << std::setw(value_width + 3)
        << throughput_host << " GB/s   |\n"
        << "+-----------------------------------------------------------------------------+"
           "\n";
}

double run_timing_host(const uint8_t *plaintext, uint8_t *key, uint8_t *nonce, uint32_t input_size,
                       uint32_t counter, uint8_t *ciphertext) {
    double total_time = 0.0;
    for (int r = 0; r < WARMUP_RUNS + NUM_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        chacha20_openssl(key, nonce, counter, plaintext, ciphertext, input_size);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        if (r >= WARMUP_RUNS) {
            total_time += elapsed;
        }
    }
    return total_time;
}

double run_timing_gpu(cl_command_queue command_queue, cl_kernel kernel, cl_mem input, cl_mem output,
                      cl_mem key, cl_mem nonce, uint32_t input_size, uint32_t counter,
                      uint8_t *output_host) {
    double total_time = 0.0;
    for (int r = 0; r < WARMUP_RUNS + NUM_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        run_kernel(command_queue, kernel, input, output, key, nonce, input_size, counter,
                   output_host);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        if (r >= WARMUP_RUNS) {
            total_time += elapsed;
        }
    }
    cleanup(0, 0, 0, 0, input, output, key, nonce);
    return total_time;
}

int main(int argc, char *argv[]) {
    unsigned long long seed = 1234ULL;
    if (argc > 1) {
        seed = std::stoull(argv[1]);
    }

    cl_context context = 0;
    cl_command_queue command_queue = 0;
    cl_program program = 0;
    cl_kernel kernel = 0;
    print_header(seed);
    std::mt19937_64 rng(seed);

    if (initialize(context, command_queue, program, kernel) != 0) {
        cleanup(context, command_queue, program, kernel, 0, 0, 0, 0);
        return 1;
    }

    for (int i = 0; i < NUM_SIZES; i++) {
        uint32_t input_size = INPUT_SIZES[i];
        uint32_t counter = 1;

        std::vector<uint8_t> plaintext(input_size);
        std::vector<uint8_t> ciphertext(input_size);
        std::vector<uint8_t> ciphertext_host(input_size);
        uint8_t key_host[CHACHA_KEY_SIZE];
        uint8_t nonce_host[CHACHA_NONCE_SIZE];
        random_bytes(plaintext.data(), input_size, rng);
        random_bytes(key_host, CHACHA_KEY_SIZE, rng);
        random_bytes(nonce_host, CHACHA_NONCE_SIZE, rng);

        cl_mem input = 0, output = 0, key = 0, nonce = 0;
        if (create_buffers(context, input_size, input, output, key, nonce, plaintext.data(),
                           key_host, nonce_host) != 0) {
            cleanup(context, command_queue, program, kernel, input, output, key, nonce);
            return 1;
        }

        double total_time_host = run_timing_host(plaintext.data(), key_host, nonce_host, input_size,
                                                 counter, ciphertext_host.data());
        double total_time_gpu = run_timing_gpu(command_queue, kernel, input, output, key, nonce,
                                               input_size, counter, ciphertext.data());

        bool match = memcmp(ciphertext.data(), ciphertext_host.data(), input_size) == 0;
        print_timing(input_size, total_time_host, total_time_gpu, match);
    }

    cleanup(context, command_queue, program, kernel, 0, 0, 0, 0);
    return 0;
}
