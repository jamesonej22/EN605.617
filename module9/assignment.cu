/**
 * @file assignment.cu
 * @author Eric Jameson
 * @brief Instantiation of GPU-based least squares solvers for equations of the form Ax = b, along
 * with helper functions and wrappers for the Module 9 assignment of EN605.617.
 */

#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>

#include <iomanip>
#include <iostream>

/** @brief The number of array sizes to use in this assignment. */
#define NUM_SIZES 6

/** @brief The collection of system sizes to solve in this assignment. The first size is used as a
 * warm-up and is not displayed in results.
 */
const int ARRAY_SIZES[NUM_SIZES] = {256, 256, 512, 1024, 2048, 4096};

/** @brief Prints the assignment header and list of matrix sizes.
 *
 * @param seed The random seed used for this run.
 */
void print_header(unsigned long long seed) {
    std::cout
        << "\n+-----------------------------------------------------------------------------+\n"
        << "| EN605.617 Module 9 Assignment                                  Eric Jameson |\n"
        << "+-----------------------------------------------------------------------------+\n"
        << "| Solving least-squares optimization problem min_x || Ax - b || for arrays of |\n"
        << "| of sizes: ";

    for (int i = 0; i < NUM_SIZES; i++) {
        if (i == 0) continue;
        std::cout << ARRAY_SIZES[i];
        if (i != NUM_SIZES - 1) {
            std::cout << ", ";
        }
    }
    std::cout
        << std::setw(42) << " |\n"
        << "|                                                                             |\n"
        << "| Random seed: " << seed << std::setw(65 - std::to_string(seed).length()) << " |\n"
        << "+-----------------------------------------------------------------------------+\n";
}

/** @brief Prints formatted timing and residual results for both solvers.
 *
 * @param n Dimension of the system.
 * @param ne_timing Normal equations runtime (ms).
 * @param ne_residual Normal equations residual.
 * @param ls_timing QR solver runtime (ms).
 * @param ls_residual QR solver residual.
 */
void print_timing_and_residuals(int n, float ne_timing, float ne_residual, float qr_timing,
                                float qr_residual) {
    int total_width = 80;
    int value_width = 17;

    std::cout << std::fixed << std::setprecision(5);
    std::cout
        << "| ARRAY SIZE: " << n << " x " << n
        << std::setw(total_width - value_width - 2 * std::to_string(n).length()) << " |\n"
        << "+-----------------------------------------------------------------------------+\n"
        // Normal equations
        << "|   Normal Equations   |                     Time Taken: " << std::setw(value_width)
        << ne_timing << " ms |\n"
        << "|                      |                     Residual:      " << std::scientific
        << std::setprecision(4) << std::setw(value_width) << ne_residual << std::fixed
        << std::setprecision(5) << " |\n"
        << "+----------------------+                                                      |\n"
        // QR
        << "|   QR Decomposition   |                     Time Taken: " << std::setw(value_width)
        << qr_timing << " ms |\n"
        << "|                      |                     Residual:      " << std::scientific
        << std::setprecision(4) << std::setw(value_width) << qr_residual << std::fixed
        << std::setprecision(5) << " |\n"
        << "+-----------------------------------------------------------------------------+\n";
}

/** @brief Generates a random square matrix of size \p n x \p n and a random vector of size \p n
 * using cuRAND.
 *
 * @param gen The cuRAND generator.
 * @param device_array Device pointer to matrix A (size \p n x \p n).
 * @param device_vector Device pointer to vector b (size \p n).
 * @param n Dimension of the system.
 */
void generate_random_data(curandGenerator_t gen, float* device_array, float* device_vector, int n) {
    curandGenerateUniform(gen, device_array, n * n);
    curandGenerateUniform(gen, device_vector, n);
}

/** @brief Solve least squares using normal equations (A^T A)x = A^T b. Note: This method is fast
 * but numerically unstable for large or ill-conditioned matrices.
 *
 * @param cublas_handle cuBLAS handle.
 * @param cusolver_handle cuSOLVER handle.
 * @param A Input matrix (size \p n x \p n).
 * @param b Input vector (size \p n)
 * @param n Dimension of the system.
 * @param[out] x Output solution vector (size \p n).
 */
void normal_equations_solver(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
                             float* A, float* b, int n, float* x) {
    float *ATA, *ATb;
    float alpha = 1.0f, beta = 0.0f;
    cudaMalloc(&ATA, n * n * sizeof(float));
    cudaMalloc(&ATb, n * sizeof(float));

    // A^T A
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, A, n, A, n, &beta, ATA,
                n);

    // A^T b
    cublasSgemv(cublas_handle, CUBLAS_OP_T, n, n, &alpha, A, n, b, 1, &beta, ATb, 1);

    // Solve (A^T A)x = A^T b using LU factorization
    int work_size = 0;
    cusolverDnSgetrf_bufferSize(cusolver_handle, n, n, ATA, n, &work_size);
    float* work_array;
    int* device_info;
    int* device_pivots;
    cudaMalloc(&work_array, work_size * sizeof(float));
    cudaMalloc(&device_info, sizeof(int));
    cudaMalloc(&device_pivots, n * sizeof(int));

    cusolverDnSgetrf(cusolver_handle, n, n, ATA, n, work_array, device_pivots, device_info);
    cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N, n, 1, ATA, n, device_pivots, ATb, n,
                     device_info);

    // Copy result and cleanup
    cudaMemcpy(x, ATb, n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(device_pivots);
    cudaFree(device_info);
    cudaFree(work_array);
    cudaFree(ATb);
    cudaFree(ATA);
}

/** @brief Solve least squares using QR decomposition. This approach is more numerically stable and
 * preferred over the normal equations approach.
 *
 * @param cublas_handle cuBLAS handle.
 * @param cusolver_handle cuSOLVER handle.
 * @param A Input matrix (size \p n x \p n).
 * @param b Input vector (size \p n)
 * @param n Dimension of the system.
 * @param[out] x Output solution vector (size \p n).
 */
void qr_decomposition_solver(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
                             float* A, float* b, int n, float* x) {
    float *work_array, *tau, *qtb;
    float alpha = 1.0f;
    int* device_info;
    cudaMalloc(&tau, n * sizeof(float));
    cudaMalloc(&qtb, n * sizeof(float));
    cudaMalloc(&device_info, sizeof(int));

    int work_size_geqrf = 0;
    cusolverDnSgeqrf_bufferSize(cusolver_handle, n, n, A, n, &work_size_geqrf);
    int work_size_ormqr = 0;
    cusolverDnSormqr_bufferSize(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, A, n, tau,
                                b, n, &work_size_ormqr);

    int work_size = std::max(work_size_geqrf, work_size_ormqr);
    cudaMalloc(&work_array, work_size * sizeof(float));

    // Copy b into qtb
    cudaMemcpy(qtb, b, n * sizeof(float), cudaMemcpyDeviceToDevice);

    // QR Factorization (A is overwritten)
    cusolverDnSgeqrf(cusolver_handle, n, n, A, n, tau, work_array, work_size, device_info);

    // Q^T b
    cusolverDnSormqr(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, A, n, tau, qtb, n,
                     work_array, work_size, device_info);

    // Solve R x = Q^T b
    cudaMemcpy(x, qtb, n * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasStrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, n, 1, &alpha, A, n, x, n);

    cudaFree(device_info);
    cudaFree(qtb);
    cudaFree(tau);
    cudaFree(work_array);
}

/** @brief Computes the residual ||Ax - b||_2.
 *
 * @param cublas_handle cuBLAS handle.
 * @param A Input matrix (size \p n x \p n).
 * @param b Input vector (size \p n).
 * @param n Dimension of the system.
 * @param x Solution vector (size \p n).
 */
float compute_residual(cublasHandle_t cublas_handle, const float* A, const float* b, int n,
                       float* x) {
    float alpha = 1.0f, beta = 0.0f, minus_one = -1.0f;
    float* tmp;
    cudaMalloc(&tmp, n * sizeof(float));

    // tmp = A * x
    cublasSgemv(cublas_handle, CUBLAS_OP_N, n, n, &alpha, A, n, x, 1, &beta, tmp, 1);

    // tmp = tmp - b
    cublasSaxpy(cublas_handle, n, &minus_one, b, 1, tmp, 1);

    float norm;
    cublasSnrm2(cublas_handle, n, tmp, 1, &norm);

    cudaFree(tmp);
    return norm;
}

/** @brief Executes a solver, measures runtime, and computes residual. Creates working copies of A
 * and b to ensure the correct residuals can be computed.
 *
 * @tparam Solver Callable solver function.
 * @param solver Solver function.
 * @param cublas_handle cuBLAS handle.
 * @param cusolver_handle cuSOLVER handle.
 * @param A Input matrix (size \p n x \p n).
 * @param b Input vector (size \p n).
 * @param n Dimension of the system.
 * @param[out] x Solution vector (size \p n).
 * @param[out] residual Computed residual value.
 * @return Execution time in milliseconds.
 */
template <typename Solver>
float solve_system(Solver solver, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
                   const float* A, const float* b, int n, float* x, float* residual) {
    float *A_copy, *b_copy;
    cudaMalloc(&A_copy, n * n * sizeof(float));
    cudaMalloc(&b_copy, n * sizeof(float));
    cudaMemcpy(A_copy, A, n * n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(b_copy, b, n * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solver(cublas_handle, cusolver_handle, A_copy, b_copy, n, x);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timing;
    cudaEventElapsedTime(&timing, start, stop);

    *residual = compute_residual(cublas_handle, A, b, n, x);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(b_copy);
    cudaFree(A_copy);
    return timing;
}

/** @brief Entry point for least-squares comparison program. For each matrix size, generates random
 * A and b, then solves the system Ax = b using both the normal equations and QR decomposition.
 * Measures runtime and residuals for both methods, then prints the results. The first iteration is
 * used as a warm-up and not reported.
 *
 * @param argc Number of arguments passed to this program.
 * @param argv Array of arguments passed to this program.
 * @return Returns 0 on success.
 */
int main(int argc, char* argv[]) {
    unsigned long long seed = 1234ULL;
    if (argc > 1) {
        seed = std::stoull(argv[1]);
    }

    print_header(seed);
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, seed);
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    for (int i = 0; i < NUM_SIZES; i++) {
        int n = ARRAY_SIZES[i];
        float *device_A, *device_b, *device_x;
        cudaMalloc(&device_A, n * n * sizeof(float));
        cudaMalloc(&device_b, n * sizeof(float));
        cudaMalloc(&device_x, n * sizeof(float));

        generate_random_data(rng, device_A, device_b, n);
        float ne_residual;
        float ne_timing = solve_system(normal_equations_solver, cublas_handle, cusolver_handle,
                                       device_A, device_b, n, device_x, &ne_residual);

        float qr_residual;
        float qr_timing = solve_system(qr_decomposition_solver, cublas_handle, cusolver_handle,
                                       device_A, device_b, n, device_x, &qr_residual);

        // First run is warmup
        if (i != 0) {
            print_timing_and_residuals(n, ne_timing, ne_residual, qr_timing, qr_residual);
        }

        cudaFree(device_x);
        cudaFree(device_b);
        cudaFree(device_A);
    }

    cusolverDnDestroy(cusolver_handle);
    cublasDestroy(cublas_handle);
    curandDestroyGenerator(rng);
}
