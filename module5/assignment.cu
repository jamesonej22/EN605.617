/**
 * @file assignment.cu
 * @author Eric Jameson
 * @brief Instantiation of GPU kernels and device functions used for AES encryption, along with
 * helper functions and wrappers for the Module 5 assignment of EN605.617.
 */

#include <omp.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "aes_cpu.hh"

/** @brief Input image file name. */
#define INPUT_FILE "jhu.ppm"
/** @brief Width of the input image in pixels. */
#define IMAGE_WIDTH 1920
/** @brief Height of the input image in pixels. */
#define IMAGE_HEIGHT 1080
/** @brief Total number of pixels in the image. */
#define IMAGE_PIXELS 1920 * 1080
/** @brief Total size of the image pixel data, accounting for 1 byte per RGB channel per pixel. */
#define IMAGE_SIZE 1920 * 1080 * 3
/** @brief Maximum value of RGB values per pixel in the PPM image format. */
#define IMAGE_MAXVAL 255
/** @brief Number of encryption iterations to perform for timing purposes. */
#define ITERATIONS 5

/** @brief Constant memory location to store the Rijndael S-Box. */
__constant__ uint8_t constant_sbox[SBOX_SIZE];
/** @brief Constant memory location to store the expanded round keys. */
__constant__ uint8_t constant_round_keys[TOTAL_KEY_SIZE];

/** @brief Global device memory location to store the Rijndael S-Box. */
__device__ uint8_t global_sbox[SBOX_SIZE];
/** @brief Globabl device memory location to store the expanded round keys. */
__device__ uint8_t global_round_keys[TOTAL_KEY_SIZE];

/**
 * @brief Helper to copy the constant memory sbox and round keys into shared memory.
 *
 * @param[out] sbox Shared memory location to store the sbox.
 * @param[out] round_keys Shared memory location to store the round keys.
 */
__device__ void aes_load_shared_memory_constant(uint8_t* sbox, uint8_t* round_keys) {
    for (int i = threadIdx.x; i < SBOX_SIZE; i += blockDim.x) {
        sbox[i] = constant_sbox[i];
    }

    for (int i = threadIdx.x; i < TOTAL_KEY_SIZE; i += blockDim.x) {
        round_keys[i] = constant_round_keys[i];
    }
}

/**
 * @brief Helper to copy the global memory sbox and round keys into shared memory.
 *
 * @param[out] sbox Shared memory location to store the sbox.
 * @param[out] round_keys Shared memory location to store the round keys.
 */
__device__ void aes_load_shared_memory_global(uint8_t* sbox, uint8_t* round_keys) {
    for (int i = threadIdx.x; i < SBOX_SIZE; i += blockDim.x) {
        sbox[i] = global_sbox[i];
    }

    for (int i = threadIdx.x; i < TOTAL_KEY_SIZE; i += blockDim.x) {
        round_keys[i] = global_round_keys[i];
    }
}

/**
 * @brief Perform the SubBytes step of encryption. For more information, see:
 * https://en.wikipedia.org/wiki/Rijndael_S-box
 *
 * @param[in,out] state The current state to operate on.
 * @param sbox Substitution table to use for SubBytes.
 */
__device__ void aes_sub_bytes(uint8_t* state, const uint8_t* sbox) {
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = sbox[state[i]];
    }
}

/**
 * @brief Perform the ShiftRows step of encryption. For more information, see:
 * https://en.wikipedia.org/wiki/Advanced_Encryption_Standard#The_ShiftRows_step
 *
 * @param[in,out] state The current state to operate on.
 */
__device__ void aes_shift_rows(uint8_t* state) {
    // Row 1: Shift left by 1
    uint8_t temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;

    // Row 2: Shift left by 2
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;

    temp = state[6];
    state[6] = state[14];
    state[14] = temp;

    // Row 3: Shift left by 3 (equivalently, right by 1)
    temp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = temp;
}

/**
 * @brief Multiply two elements of GF(2^8). Adapted from
 * https://en.wikipedia.org/wiki/Rijndael_MixColumns#Implementation_example
 *
 * @param a The first multiplicand.
 * @param b The second multiplicand.
 * @return The product of \p a and \p b in GF(2^8).
 */
__device__ uint8_t aes_galois_multiplication(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (int i = 0; i < BITS_PER_BYTE; i++) {
        if ((b & 1) != 0) {
            p ^= a;
        }

        bool high_bit_set = (a & 0x80) != 0;
        a <<= 1;
        if (high_bit_set) {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    return p;
}

/**
 * @brief Perform the MixColumns step of encryption. For more information, see:
 * https://en.wikipedia.org/wiki/Rijndael_MixColumns
 *
 * @param[in,out] state The current state to operate on.
 */
__device__ void aes_mix_columns(uint8_t* state) {
    for (int i = 0; i < STATE_DIMENSION; i++) {
        int start_idx = i * STATE_DIMENSION;
        uint8_t s0 = state[start_idx];
        uint8_t s1 = state[start_idx + 1];
        uint8_t s2 = state[start_idx + 2];
        uint8_t s3 = state[start_idx + 3];

        state[start_idx] =
            aes_galois_multiplication(0x02, s0) ^ aes_galois_multiplication(0x03, s1) ^ s2 ^ s3;
        state[start_idx + 1] =
            s0 ^ aes_galois_multiplication(0x02, s1) ^ aes_galois_multiplication(0x03, s2) ^ s3;
        state[start_idx + 2] =
            s0 ^ s1 ^ aes_galois_multiplication(0x02, s2) ^ aes_galois_multiplication(0x03, s3);
        state[start_idx + 3] =
            aes_galois_multiplication(0x03, s0) ^ s1 ^ s2 ^ aes_galois_multiplication(0x02, s3);
    }
}

/**
 * @brief Perform the AddRoundKey step of encryption. For more information, see:
 * https://en.wikipedia.org/wiki/Advanced_Encryption_Standard#The_AddRoundKey
 *
 * @param[in,out] state The current state to operate on.
 * @param round_keys The expanded round keys used for encryption.
 * @param round The round of encryption that we are on.
 */
__device__ void aes_add_round_key(uint8_t* state, const uint8_t* round_keys, int round) {
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] ^= round_keys[round * KEY_SIZE + i];
    }
}

/**
 * @brief Perform the entirety of AES encryption on the input data using global memory arrays.
 *
 * @param input Plaintext to encrypt.
 * @param[out] output Location to store the encrypted ciphertext.
 * @param round_keys Expanded keys to use during encryption.
 * @param blocks_to_encrypt Number of 16-byte blocks to encrypt. Assumed to be the size of the
 * input data divided by 16.
 */
__global__ void aes_encrypt_global(const uint8_t* input, uint8_t* output, int blocks_to_encrypt) {
    int grid_size = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < blocks_to_encrypt; idx += grid_size) {
        uint8_t state[STATE_SIZE];
        for (int i = 0; i < STATE_SIZE; i++) {
            state[i] = input[idx * STATE_SIZE + i];
        }

        aes_add_round_key(state, global_round_keys, 0);

        for (int round = 1; round < NUM_ROUNDS; round++) {
            aes_sub_bytes(state, global_sbox);
            aes_shift_rows(state);
            aes_mix_columns(state);
            aes_add_round_key(state, global_round_keys, round);
        }

        aes_sub_bytes(state, global_sbox);
        aes_shift_rows(state);
        aes_add_round_key(state, global_round_keys, NUM_ROUNDS);

        for (int i = 0; i < STATE_SIZE; i++) {
            output[idx * STATE_SIZE + i] = state[i];
        }
    }
}

/**
 * @brief Perform the entirety of AES encryption on the input data using constant memory arrays.
 *
 * @param input Plaintext to encrypt.
 * @param[out] output Location to store the encrypted ciphertext.
 * @param round_keys Expanded keys to use during encryption.
 * @param blocks_to_encrypt Number of 16-byte blocks to encrypt. Assumed to be the size of the
 * input data divided by 16.
 */
__global__ void aes_encrypt_constant(const uint8_t* input, uint8_t* output, int blocks_to_encrypt) {
    int grid_size = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < blocks_to_encrypt; idx += grid_size) {
        uint8_t state[STATE_SIZE];
        for (int i = 0; i < STATE_SIZE; i++) {
            state[i] = input[idx * STATE_SIZE + i];
        }

        aes_add_round_key(state, constant_round_keys, 0);

        for (int round = 1; round < NUM_ROUNDS; round++) {
            aes_sub_bytes(state, constant_sbox);
            aes_shift_rows(state);
            aes_mix_columns(state);
            aes_add_round_key(state, constant_round_keys, round);
        }

        aes_sub_bytes(state, constant_sbox);
        aes_shift_rows(state);
        aes_add_round_key(state, constant_round_keys, NUM_ROUNDS);

        for (int i = 0; i < STATE_SIZE; i++) {
            output[idx * STATE_SIZE + i] = state[i];
        }
    }
}

/**
 * @brief Perform the entirety of AES encryption on the input data using shared memory arrays,
 * copied from constant memory.
 *
 * @param input Plaintext to encrypt.
 * @param[out] output Location to store the encrypted ciphertext.
 * @param round_keys Expanded keys to use during encryption.
 * @param blocks_to_encrypt Number of 16-byte blocks to encrypt. Assumed to be the size of the
 * input data divided by 16.
 */
__global__ void aes_encrypt_shared_from_constant(const uint8_t* input, uint8_t* output,
                                                 int blocks_to_encrypt) {
    __shared__ uint8_t shared_sbox[SBOX_SIZE];
    __shared__ uint8_t shared_round_keys[TOTAL_KEY_SIZE];

    aes_load_shared_memory_constant(shared_sbox, shared_round_keys);
    __syncthreads();

    int grid_size = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < blocks_to_encrypt; idx += grid_size) {
        uint8_t state[STATE_SIZE];
        for (int i = 0; i < STATE_SIZE; i++) {
            state[i] = input[idx * STATE_SIZE + i];
        }

        aes_add_round_key(state, shared_round_keys, 0);

        for (int round = 1; round < NUM_ROUNDS; round++) {
            aes_sub_bytes(state, shared_sbox);
            aes_shift_rows(state);
            aes_mix_columns(state);
            aes_add_round_key(state, shared_round_keys, round);
        }

        aes_sub_bytes(state, shared_sbox);
        aes_shift_rows(state);
        aes_add_round_key(state, shared_round_keys, NUM_ROUNDS);

        for (int i = 0; i < STATE_SIZE; i++) {
            output[idx * STATE_SIZE + i] = state[i];
        }
    }
}

/**
 * @brief Perform the entirety of AES encryption on the input data using shared memory arrays,
 * copied from global memory.
 *
 * @param input Plaintext to encrypt.
 * @param[out] output Location to store the encrypted ciphertext.
 * @param round_keys Expanded keys to use during encryption.
 * @param blocks_to_encrypt Number of 16-byte blocks to encrypt. Assumed to be the size of the
 * input data divided by 16.
 */
__global__ void aes_encrypt_shared_from_global(const uint8_t* input, uint8_t* output,
                                               int blocks_to_encrypt) {
    __shared__ uint8_t shared_sbox[SBOX_SIZE];
    __shared__ uint8_t shared_round_keys[TOTAL_KEY_SIZE];

    aes_load_shared_memory_global(shared_sbox, shared_round_keys);
    __syncthreads();

    int grid_size = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < blocks_to_encrypt; idx += grid_size) {
        uint8_t state[STATE_SIZE];
        for (int i = 0; i < STATE_SIZE; i++) {
            state[i] = input[idx * STATE_SIZE + i];
        }

        aes_add_round_key(state, shared_round_keys, 0);

        for (int round = 1; round < NUM_ROUNDS; round++) {
            aes_sub_bytes(state, shared_sbox);
            aes_shift_rows(state);
            aes_mix_columns(state);
            aes_add_round_key(state, shared_round_keys, round);
        }

        aes_sub_bytes(state, shared_sbox);
        aes_shift_rows(state);
        aes_add_round_key(state, shared_round_keys, NUM_ROUNDS);

        for (int i = 0; i < STATE_SIZE; i++) {
            output[idx * STATE_SIZE + i] = state[i];
        }
    }
}

/** @brief Helper to print the header with thread and block information.
 *
 * @param total_threads Requested (or adjusted) number of total CUDA threads
 * @param cpu_threads Best approximation to the number of CPU threads running in this program.
 * @param block_size Requested number of threads per block
 * @param num_blocks Requested number of blocks
 * @param warning Flag to indicated if the number of total threads has been adjusted to
 * accomodate the requested block_size
 */
void print_header(int cuda_threads, int cpu_threads, int block_size, int num_blocks, bool warning) {
    std::cout << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl
              << "| EN605.617 Module 5 Assignment                                  Eric Jameson |"
              << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl;

    std::cout << "| Total CUDA Threads:"
              << std::setw(80 - 24 - std::to_string(cuda_threads).length()) << " " << cuda_threads
              << " |" << std::endl
              << "| Total CPU Threads (best guess):"
              << std::setw(80 - 36 - std::to_string(cpu_threads).length()) << " " << cpu_threads
              << " |" << std::endl
              << "| Block Size:       " << std::setw(80 - 23 - std::to_string(block_size).length())
              << " " << block_size << " |" << std::endl
              << "| Number of Blocks: " << std::setw(80 - 23 - std::to_string(num_blocks).length())
              << " " << num_blocks << " |" << std::endl;

    if (warning) {
        std::cout
            << "| " << std::setw(77) << " |" << std::endl
            << "| Warning: Chosen thread count is not evenly divisible by the block size, so  |"
            << std::endl
            << "| the total number of threads has been rounded up to " << cuda_threads << "."
            << std::setw(80 - 55 - std::to_string(cuda_threads).length()) << " |" << std::endl;
    }

    int chunks_per_thread = IMAGE_SIZE / STATE_SIZE / cuda_threads;
    if (chunks_per_thread == 0) {
        chunks_per_thread++;
    }
    std::string chunk = chunks_per_thread == 1 ? "chunk" : "chunks";
    std::cout << "|                                                                             |"
              << std::endl
              << "| Each thread will encrypt " << chunks_per_thread << " plaintext " << chunk
              << " of size " << STATE_SIZE << "."
              << std::setw(80 - 49 - std::to_string(chunks_per_thread).length() - chunk.length() -
                           std::to_string(STATE_SIZE).length())
              << " |" << std::endl
              << "| Running " << ITERATIONS << " iterations of each type. "
              << std::setw(80 - 37 - std::to_string(ITERATIONS).length()) << " |" << std::endl;

    std::cout << "+-----------------------------------------------------------------------------+"
              << std::endl;
}

/**
 * @brief Wrapper for running and timing the CPU-only implementation of AES.
 *
 * @param plaintext The plaintext to encrypt.
 * @param[out] ciphertext The output ciphertext.
 * @param plaintext_length The length of the plaintext in bytes.
 * @param expanded_key The expanded key to use for encryption.
 * @return Average time taken for encryption over ITERATIONS iterations.
 */
float run_aes_cpu(const uint8_t* plaintext, uint8_t* ciphertext, size_t plaintext_length,
                  const uint8_t* expanded_key) {
    size_t blocks_to_encrypt = plaintext_length / STATE_SIZE;
    float time = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        aes_encrypt_cpu(plaintext, ciphertext, expanded_key, blocks_to_encrypt);
        auto stop = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration<float, std::milli>(stop - start).count();
    }
    return time / ITERATIONS;
}

/**
 * @brief Wrapping for running and timing the various GPU implementations of AES.
 *
 * @tparam AESKernel Generic kernel type used to abstract out the specific kernel.
 * @param kernel The encryption kernel to run.
 * @param num_blocks The number of CUDA thread blocks to use
 * @param block_size The number of CUDA threads per block
 * @param plaintext The plaintext to encrypt.
 * @param[out] ciphertext The output ciphertext.
 * @param plaintext_length The length of the plaintext in bytes.
 * @return Average time taken for encryption over ITERATIONS iterations.
 */
template <typename AESKernel>
float run_aes_kernel(AESKernel kernel, int num_blocks, int block_size, const uint8_t* plaintext,
                     uint8_t* ciphertext, size_t plaintext_length) {
    // Set up timing and memory
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint8_t* device_pt;
    uint8_t* device_ct;
    size_t blocks_to_encrypt = plaintext_length / STATE_SIZE;
    cudaMalloc(&device_pt, blocks_to_encrypt * STATE_SIZE);
    cudaMalloc(&device_ct, blocks_to_encrypt * STATE_SIZE);
    cudaMemcpy(device_pt, plaintext, plaintext_length, cudaMemcpyHostToDevice);

    // Dry run (no timing)
    kernel<<<num_blocks, block_size>>>(device_pt, device_ct, blocks_to_encrypt);
    cudaDeviceSynchronize();
    float time = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start);
        kernel<<<num_blocks, block_size>>>(device_pt, device_ct, blocks_to_encrypt);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        time += elapsed_time;
    }

    // Copy ciphertext back to host
    cudaMemcpy(ciphertext, device_ct, plaintext_length, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_ct);
    cudaFree(device_pt);
    return time / ITERATIONS;
}

/**
 * @brief Helper to read a PPM image from INPUT_FILE.
 *
 * @return The pixel data of the image (excluding the header) as a vector of bytes.
 */
std::vector<uint8_t> read_image_from_file() {
    std::ifstream in(INPUT_FILE, std::ios::binary);
    std::string magic;
    int width, height, maxval;

    // Read header
    in >> magic;
    char c;
    in >> width >> height >> maxval;
    in.get(c);

    // Read pixel data
    const size_t image_size = width * height * 3;
    std::vector<uint8_t> pixels(image_size);
    in.read(reinterpret_cast<char*>(pixels.data()), image_size);
    return pixels;
}

/**
 * @brief Helper function to print timing information about the given encryption method.
 *
 * @param time The average time taken to encrypt using this method.
 * @param encryption_type Name of the encryption method.
 */
void print_timing_results(float time, std::string encryption_type) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << time;
    std::string time_str = oss.str();

    std::cout << "| " << encryption_type << std::setw(80 - 3 - encryption_type.length()) << " |"
              << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl
              << "| Average time taken: " << std::setw(80 - 28 - time_str.length()) << " "
              << time_str << " ms |" << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl;
}

/**
 * @brief Utility to write a vector of PPM pixel data to a file. Assumes that the pixel data is
 * of the correct size.
 *
 * @param data Reference to the vector of pixel data.
 * @param filename Output filename.
 */
void write_image_to_file(std::vector<uint8_t>& data, std::string filename) {
    std::ofstream out_device(filename, std::ios::binary);
    out_device << "P6\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n" << IMAGE_MAXVAL << "\n";
    out_device.write(reinterpret_cast<const char*>(data.data()), data.size());
}

/**
 * @brief Wrapper around all GPU kernels used for this assignment. Copies sbox and expanded key to
 * global or constant memory as required, and then runs each kernel, storing the output image and
 * printing timing data.
 *
 * @param num_blocks The number of CUDA thread blocks to use.
 * @param block_size The number of CUDA threads per block.
 * @param plaintext Reference to the plaintext data to encrypt.
 * @param ciphertext Reference to the output vector for the encrypted ciphertext.
 * @param expanded_key The expanded key to use for encryption.
 */
void run_gpu_kernels(int num_blocks, int block_size, const std::vector<uint8_t>& plaintext,
                     std::vector<uint8_t>& ciphertext, const uint8_t* expanded_key) {
    // Encrypt using global memory
    cudaMemcpyToSymbol(global_sbox, sbox, SBOX_SIZE);
    cudaMemcpyToSymbol(global_round_keys, expanded_key, TOTAL_KEY_SIZE);
    float time = run_aes_kernel(aes_encrypt_global, num_blocks, block_size, plaintext.data(),
                                ciphertext.data(), plaintext.size());

    write_image_to_file(ciphertext, "encrypted___global.ppm");
    print_timing_results(time, "Global memory");

    // Encrypt using constant memory
    cudaMemcpyToSymbol(constant_sbox, sbox, SBOX_SIZE);
    cudaMemcpyToSymbol(constant_round_keys, expanded_key, TOTAL_KEY_SIZE);
    time = run_aes_kernel(aes_encrypt_constant, num_blocks, block_size, plaintext.data(),
                          ciphertext.data(), plaintext.size());

    write_image_to_file(ciphertext, "encrypted_constant.ppm");
    print_timing_results(time, "Constant memory");

    // Encrypt using shared memory copied from constant
    time = run_aes_kernel(aes_encrypt_shared_from_constant, num_blocks, block_size,
                          plaintext.data(), ciphertext.data(), plaintext.size());

    write_image_to_file(ciphertext, "encrypted_shared_c.ppm");
    print_timing_results(time, "Shared memory (from constant)");

    // Encrypt using shared memory copied from global
    time = run_aes_kernel(aes_encrypt_shared_from_global, num_blocks, block_size, plaintext.data(),
                          ciphertext.data(), plaintext.size());

    write_image_to_file(ciphertext, "encrypted_shared_g.ppm");
    print_timing_results(time, "Shared memory (from global)");
}

/**
 * @brief Main function for the module 5 assignment. Handles command-line input, reads input
 * image, and runs key expansion on the constant key. Then calls out to run CPU-only encryption
 * and the various GPU kernels to perform the same encryption. Displays timing data for
 * each part.
 *
 * @param argc Number of arguments passed to this program
 * @param argv Array of arguments passed to this program
 * @return Returns 0 on success.
 */
int main(int argc, char* argv[]) {
    // Handle command-line input and print header
    int total_threads = IMAGE_SIZE / STATE_SIZE;
    int block_size = 192;

    if (argc >= 2) {
        total_threads = atoi(argv[1]);
    }
    if (argc >= 3) {
        block_size = atoi(argv[2]);
    }

    bool warning = false;
    int num_blocks = total_threads / block_size;
    if (total_threads % block_size != 0) {
        num_blocks++;
        total_threads = num_blocks * block_size;
        warning = true;
    }
    int cpu_threads = omp_get_num_procs();

    print_header(total_threads, cpu_threads, block_size, num_blocks, warning);
    uint8_t expanded_key[TOTAL_KEY_SIZE];
    aes_expand_key(expanded_key, key);

    std::vector<uint8_t> plaintext = read_image_from_file();
    std::vector<uint8_t> ciphertext(plaintext.size());

    // Encrypt using host memory
    float time = run_aes_cpu(plaintext.data(), ciphertext.data(), plaintext.size(), expanded_key);
    write_image_to_file(ciphertext, "encrypted_host.ppm");
    print_timing_results(time, "Host memory");

    run_gpu_kernels(num_blocks, block_size, plaintext, ciphertext, expanded_key);
    return 0;
}