/**
 * @file assignment.cu
 * @author Eric Jameson
 * @brief Implementation of wrapper and helper functions used for AES encryption for the Module 6
 * assignment of EN605.617.
 */

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "aes.cuh"

/** @brief Input image file name. */
#define INPUT_FILE "cuda.ppm"
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
#define ITERATIONS 10
/** @brief Maximum number of CUDA streams to use in this assignment */
#define MAX_STREAMS 16

/** @brief Helper to print the header with thread and block information.
 *
 * @param total_threads Requested (or adjusted) number of total CUDA threads
 * @param block_size Requested number of threads per block
 * @param num_blocks Requested number of blocks
 * @param warning Flag to indicated if the number of total threads has been adjusted to
 * accomodate the requested block_size
 */
void print_header(int cuda_threads, int block_size, int num_blocks, bool warning) {
    std::cout << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl
              << "| EN605.617 Module 6 Assignment                                  Eric Jameson |"
              << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl;

    std::cout << "| Total CUDA Threads:"
              << std::setw(80 - 24 - std::to_string(cuda_threads).length()) << " " << cuda_threads
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
 * @brief Wrapper for running and stream-accelerated GPU AES kernels.
 *
 * @tparam AESKernel Generic kernel type used to abstract out the specific AES kernel to run.
 * @param kernel The encryption kernel to run.
 * @param num_blocks The number of CUDA thread blocks to use.
 * @param block_size The number of CUDA threads per block.
 * @param num_streams The number of parallel CUDA streams to use.
 * @param streams Array of actual CUDA streams to use for encryption.
 * @param device_pt Pre-allocated device plaintext memory location, broken into chunks for optimal
 * stream accessibility.
 * @param[out] device_ct Pre-allocated device ciphertext memory location, similarly broken into
 * chunks for stream accessibility.
 * @param host_pt Host plaintext memory location.
 * @param[out] host_ct Host ciphertext memory location
 * @param plaintext_length Length of the plaintext in bytes.
 */
template <typename AESKernel>
void run_stream_pipeline(AESKernel kernel, int num_blocks, int block_size, int num_streams,
                         cudaStream_t* streams, uint8_t** device_pt, uint8_t** device_ct,
                         uint8_t* host_pt, uint8_t* host_ct, size_t plaintext_length) {
    size_t chunk_size = (plaintext_length + num_streams - 1) / num_streams;

    for (int s = 0; s < num_streams; s++) {
        size_t offset = s * chunk_size;
        size_t block_offset = offset / 16;

        if (offset >= plaintext_length) break;

        size_t current_size = std::min(chunk_size, plaintext_length - offset);

        cudaMemcpyAsync(device_pt[s], host_pt + offset, current_size, cudaMemcpyHostToDevice,
                        streams[s]);
        kernel<<<num_blocks, block_size, 0, streams[s]>>>(device_pt[s], device_ct[s], current_size,
                                                          block_offset);
        cudaMemcpyAsync(host_ct + offset, device_ct[s], current_size, cudaMemcpyDeviceToHost,
                        streams[s]);
    }
}

/**
 * @brief Wrapper for running and timing the various GPU AES kernels.
 *
 * @tparam AESKernel Generic kernel type used to abstract out the specific AES kernel to run.
 * @param kernel The encryption kernel to run.
 * @param num_blocks The number of CUDA thread blocks to use.
 * @param block_size The number of CUDA threads per block.
 * @param num_streams The number of parallel CUDA streams to use.
 * @param plaintext The plaintext to encrypt.
 * @param[out] ciphertext The output ciphertext.
 * @param plaintext_length The length of the plaintext in bytes.
 * @return Average time taken for encryption over ITERATIONS iterations.
 */
template <typename AESKernel>
float run_aes_kernel(AESKernel kernel, int num_blocks, int block_size, int num_streams,
                     const uint8_t* plaintext, uint8_t* ciphertext, size_t plaintext_length) {
    // Set up timing and memory
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    uint8_t* host_pt;
    uint8_t* host_ct;
    cudaHostAlloc(&host_pt, plaintext_length, cudaHostAllocDefault);
    cudaHostAlloc(&host_ct, plaintext_length, cudaHostAllocDefault);
    memcpy(host_pt, plaintext, plaintext_length);

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate device buffers per stream
    std::vector<uint8_t*> device_pt(num_streams);
    std::vector<uint8_t*> device_ct(num_streams);
    size_t chunk_size = (plaintext_length + num_streams - 1) / num_streams;
    for (int i = 0; i < num_streams; i++) {
        cudaMalloc(&device_pt[i], chunk_size);
        cudaMalloc(&device_ct[i], chunk_size);
    }

    // Dry run (no timing)
    run_stream_pipeline(kernel, num_blocks, block_size, num_streams, streams.data(),
                        device_pt.data(), device_ct.data(), host_pt, host_ct, plaintext_length);
    cudaDeviceSynchronize();

    // Timed runs
    float total_time = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start);
        run_stream_pipeline(kernel, num_blocks, block_size, num_streams, streams.data(),
                            device_pt.data(), device_ct.data(), host_pt, host_ct, plaintext_length);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time += elapsed;
    }
    cudaDeviceSynchronize();
    memcpy(ciphertext, host_ct, plaintext_length);

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaFree(device_pt[i]);
        cudaFree(device_ct[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(host_pt);
    cudaFreeHost(host_ct);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_time / ITERATIONS;
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
void print_timing_results(std::vector<float> timings, std::vector<int> num_streams,
                          std::string encryption_type) {
    std::cout << "| " << encryption_type << std::setw(80 - 3 - encryption_type.length()) << " |"
              << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl;

    for (size_t i = 0; i < num_streams.size(); i++) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << timings[i];
        std::string time_str = oss.str();

        std::cout << "| Number of streams: " << std::setw(3) << num_streams[i]
                  << "  |  Average time taken: " << std::setw(53 - 28 - time_str.length()) << " "
                  << time_str << " ms |" << std::endl;
    }

    std::cout << "+-----------------------------------------------------------------------------+"
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
 * @brief Wrapper around all GPU kernels used for this assignment. Copies sbox and expanded key
 * to global or constant memory as required, and then runs each kernel, storing the output image
 * and printing timing data.
 *
 * @param num_blocks The number of CUDA thread blocks to use.
 * @param block_size The number of CUDA threads per block.
 * @param plaintext Reference to the plaintext data to encrypt.
 * @param ciphertext Reference to the output vector for the encrypted ciphertext.
 * @param expanded_key The expanded key to use for encryption.
 */
void run_gpu_kernels(int num_blocks, int block_size, const std::vector<uint8_t>& plaintext,
                     std::vector<uint8_t>& ciphertext, const uint8_t* expanded_key) {
    cudaMemcpyToSymbol(device_sbox, sbox, SBOX_SIZE);
    cudaMemcpyToSymbol(device_round_keys, expanded_key, TOTAL_KEY_SIZE);
    float time;

    // Encrypt using ECB mode
    std::vector<float> ecb_timings;
    std::vector<int> ecb_streams;

    for (int i = 1; i <= MAX_STREAMS; i *= 2) {
        time = run_aes_kernel(aes_encrypt_ecb_kernel, num_blocks, block_size, i, plaintext.data(),
                              ciphertext.data(), plaintext.size());
        ecb_timings.push_back(time);
        ecb_streams.push_back(i);

        std::ostringstream filename;
        filename << "encrypted_ecb_" << i << ".ppm";

        write_image_to_file(ciphertext, filename.str());
    }

    print_timing_results(ecb_timings, ecb_streams, "ECB Mode");

    // Encrypt using CTR mode
    std::vector<float> ctr_timings;
    std::vector<int> ctr_streams;

    for (int i = 1; i <= MAX_STREAMS; i *= 2) {
        time = run_aes_kernel(aes_encrypt_ctr_kernel, num_blocks, block_size, i, plaintext.data(),
                              ciphertext.data(), plaintext.size());
        ctr_timings.push_back(time);
        ctr_streams.push_back(i);

        std::ostringstream filename;
        filename << "encrypted_ctr_" << i << ".ppm";

        write_image_to_file(ciphertext, filename.str());
    }

    print_timing_results(ctr_timings, ctr_streams, "CTR Mode");
}

/**
 * @brief Main function for the module 6 assignment. Handles command-line input, reads input
 * image, and runs key expansion on the constant key. Calls out to run AES-ECB encryption
 * and AES-CTR encryption on the image, and finally displays timing data for all kernel runs.
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

    print_header(total_threads, block_size, num_blocks, warning);
    uint8_t expanded_key[TOTAL_KEY_SIZE];
    aes_expand_key(expanded_key, key);

    std::vector<uint8_t> plaintext = read_image_from_file();
    std::vector<uint8_t> ciphertext(plaintext.size());

    run_gpu_kernels(num_blocks, block_size, plaintext, ciphertext, expanded_key);
    return 0;
}