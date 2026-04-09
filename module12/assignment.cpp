/** @file assignment.cpp
 * @author Eric Jameson
 * @brief Functions and helpers to set-up OpenCL-based ChaCha20 encryption of multiple random files
 * with comparison against OpenSSL reference implementations for the Module 12 assignment of
 * EN605.617.
 */

/** @brief Target version of OpenCL. */
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

/** @brief The number of input sizes. */
#define NUM_SIZES 4
/** @brief The number of timing runs per input size. */
#define NUM_RUNS 5
/** @brief The number of warmup (non-timing) runs per input size. */
#define WARMUP_RUNS 1
/** @brief The size of a ChaCha key in bytes. */
#define CHACHA_KEY_SIZE 32
/** @brief The size of a ChaCha nonce in bytes. */
#define CHACHA_NONCE_SIZE 12

/** @brief The input sizes to use for timing comparisons. */
const uint32_t INPUT_SIZES[NUM_SIZES] = {1 << 20, 1 << 24, 1 << 28, 1 << 30};

/** @brief Wrapper for OpenSSL-based reference implementation of ChaCha20 for correctness
 * verification.
 *
 * @param key The encryption key for this run of ChaCha20.
 * @param nonce The nonce for this run of ChaCha20.
 * @param counter The initial counter for encryption.
 * @param input The plaintext to encrypt or ciphertext to decrypt.
 * @param[out] output The encrypted plaintext or decrypted ciphertext.
 * @param len The length of the input in bytes.
 */
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

/** @brief Helper to generate random byte arrays of a particular size.
 *
 * @param[out] data The location to store the random bytes.
 * @param size The number of bytes to generate.
 * @param rng Previously initialized random number generator.
 */
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

/** @brief OpenCL helper to create a GPU or CPU context after finding platforms.
 *
 * @return NULL if no context can be created, otherwise the initialized context.
 */
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

/** @brief OpenCL helper to create a command queue using the previously initialized context.
 *
 * @param context The previously initialized OpenCL context.
 * @param[out] device The first initialized device.
 * @return NULL if no command queue can be created, otherwise the initialized command queue.
 */
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

/** @brief OpenCL helper to create a program using a previously initialized context and device.
 *
 * @param context The previously initialized OpenCL context.
 * @param device The previously initialized OpenCL device.
 * @param filename The name of the file containing the OpenCL kernel.
 * @return NULL if no program can be created, otherwise the created OpenCL program.
 */
cl_program create_program(cl_context context, cl_device_id device, const char *filename) {
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(filename, std::ios::in);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
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

/** @brief Wrapper to create all necessary pieces for OpenCL operation in this assignment. Inputs
 * are unitialized and passed by reference so that they can be initialized in their respective
 * initialization functions.
 *
 * @param[in,out] context The OpenCL context to intialize.
 * @param[in,out] command_queue The OpenCL command queue to initialize.
 * @param[in,out] program The OpenCL program to initialize.
 * @param[in,out] kernel The OpenCL kernel to initialize.
 * @return 1 if any piece fails to initialize properly, 0 otherwise.
 */
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

/** @brief OpenCL helper to initialize all memory buffers for this program, copying memory from the
 * host if applicable.
 *
 * @param context The initialized OpenCL context.
 * @param input_size The size of the plaintext in bytes.
 * @param input The OpenCL memory buffer for the input.
 * @param output The OpenCL memory buffer for the output.
 * @param key The OpenCL memory buffer for the key.
 * @param nonce The OpenCL memory buffer for the nonce.
 * @param plaintext The host location for the plaintext array.
 * @param key_host The host location for the key.
 * @param nonce_host The host location for the nonce.
 * @return 1 if any buffer is not created successfully, 0 otherwise.
 */
int create_buffers(cl_context context, uint32_t input_size, cl_mem &input, cl_mem &output,
                   cl_mem &key, cl_mem &nonce, const uint8_t *plaintext, const uint8_t *key_host,
                   const uint8_t *nonce_host) {
    input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size,
                           (void *)plaintext, NULL);

    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input_size, NULL, NULL);

    key = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, (void *)key_host,
                         NULL);

    nonce = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 12, (void *)nonce_host,
                           NULL);

    if (!input || !output || !key || !nonce) {
        std::cerr << "Error creating buffers\n";
        return 1;
    }

    return 0;
}

/** @brief Wrapper to run the OpenCL ChaCha20 program with previously initialized command queue,
 * kernel, and memory objects. Stores the output back into a host buffer for ease of use.
 *
 * @param command_queue The initialized OpenCL command queue.
 * @param kernel The initialized OpenCL kernel.
 * @param input The OpenCL memory buffer for the input.
 * @param output The OpenCL memory buffer for the output.
 * @param key The OpenCL memory buffer for the key.
 * @param nonce The OpenCL memory buffer for the nonce.
 * @param input_size The size of the plaintext in bytes.
 * @param counter The initial counter for encryption.
 * @param[out] ciphertext The host location to store the ciphertext.
 * @return 1 if any errors occur in running the kernel, 0 otherwise.
 */
int run_kernel(cl_command_queue command_queue, cl_kernel kernel, cl_mem input, cl_mem output,
               cl_mem key, cl_mem nonce, uint32_t input_size, uint32_t counter,
               uint8_t *ciphertext) {
    cl_int errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &key);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &nonce);
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

/** @brief Helper to release all OpenCL objects to ensure there are no memory leaks.
 *
 * @param context The OpenCL context object to release.
 * @param command_queue The OpenCL command queue object to release.
 * @param program The OpenCL program object to release.
 * @param kernel The OpenCL kernel object to release.
 * @param input The input OpenCL memory object to release.
 * @param output The output OpenCL memory object to release.
 * @param key The key OpenCL memory object to release.
 * @param nonce The nonce OpenCL memory object to release.
 */
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

/** @brief Helper to print the assignment header and random seed, along with input sizes that
 * will be measured.
 *
 * @param seed The random seed used for this program.
 */
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

/** @brief Helper to print the timing results for the given input size, host/device times, and an
 * indication on whether the outputs match.
 *
 * @param input_size The input size in bytes.
 * @param total_time_host The total time taken for all runs on the host.
 * @param total_time_gpu The total time taken for all runs on the device.
 * @param match Indication of whether the outputs for this run match.
 */
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

/** @brief Wrapper to call the OpenSSL reference implementation of ChaCha20 and time how long it
 * takes.
 *
 * @param plaintext The plaintext to encrypt.
 * @param key The encryption key.
 * @param nonce The nonce used for encryption.
 * @param input_size The input size in bytes.
 * @param counter The initial counter for encryption.
 * @param[out] ciphertext Location to store the ciphertext.
 * @return The total time taken for all runs (excluding warmup runs).
 */
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

/** @brief Wrapper to call the OpenCL implementation of ChaCha20 and time how long it takes.
 *
 * @param command_queue The OpenCL command queue object.
 * @param kernel The OpenCL kernel object.
 * @param input The input OpenCL memory object.
 * @param output The output OpenCL memory object.
 * @param key The key OpenCL memory object.
 * @param nonce The nonce OpenCL memory object.
 * @param input_size The input size in bytes.
 * @param counter The initial counter for encryption.
 * @param[out] output_host The location to store the ciphertext on the host.
 * @return The total time taken for all runs (excluding warmup runs).
 */
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

/** @brief Entry point for the ChaCha20 encryption program. For each input size, generates random
 * plaintext bytes, then encrypts the plaintext using both the OpenCL implementation found in
 * chacha20.cl and an OpenSSL reference implementaiton. Measures runtime for both methods, compares
 * the outputs, and then prints the results.
 *
 * @param argc Number of arguments passed to this program.
 * @param argv Array of arguments passed to this program.
 * @return 1 if there are any errors in OpenCL instantiation, 0 otherwise.
 */
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
