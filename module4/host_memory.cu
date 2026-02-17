#include <stdio.h>

// From https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/

__global__ void saxpy(int n, float a, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(void) {
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float h2d_time, kernel_time, d2h_time;

    //-----------------------------------------------------------------------//
    // Pageable Memory                                                       //
    //-----------------------------------------------------------------------//
    printf("=== Pageable Memory ===\n");

    // Time H->D transfer
    cudaEventRecord(start);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_time, start, stop);

    // Time kernel execution
    cudaEventRecord(start);
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Time D->H transfer
    cudaEventRecord(start);
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    printf("  Average H->D time: %.3f μs\n", h2d_time * 1000 / 2);
    printf("  Kernel time: %.3f μs\n", kernel_time * 1000);
    printf("  D->H time: %.3f μs\n", d2h_time * 1000);
    printf("  Total time: %.3f μs\n\n", (h2d_time + kernel_time + d2h_time) * 1000);

    //-----------------------------------------------------------------------//
    // Pinned Memory                                                         //
    //-----------------------------------------------------------------------//
    printf("=== Pinned Memory ===\n");

    float *x_pinned, *y_pinned;
    cudaMallocHost(&x_pinned, N * sizeof(float));
    cudaMallocHost(&y_pinned, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x_pinned[i] = 1.0f;
        y_pinned[i] = 2.0f;
    }

    // Time H->D transfer
    cudaEventRecord(start);
    cudaMemcpy(d_x, x_pinned, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_pinned, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_time, start, stop);

    // Time kernel execution
    cudaEventRecord(start);
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Time D->H transfer
    cudaEventRecord(start);
    cudaMemcpy(y_pinned, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    printf("  Average H->D time: %.3f μs\n", h2d_time * 1000 / 2);
    printf("  Kernel time: %.3f μs\n", kernel_time * 1000);
    printf("  D->H time: %.3f μs\n", d2h_time * 1000);
    printf("  Total time: %.3f μs\n\n", (h2d_time + kernel_time + d2h_time) * 1000);

    cudaFreeHost(x_pinned);
    cudaFreeHost(y_pinned);

    //-----------------------------------------------------------------------//
    // Mapped Memory                                                         //
    //-----------------------------------------------------------------------//
    printf("=== Mapped Memory ===\n");

    float *x_mapped, *y_mapped, *d_x_mapped, *d_y_mapped;
    cudaHostAlloc(&x_mapped, N * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&y_mapped, N * sizeof(float), cudaHostAllocMapped);

    cudaHostGetDevicePointer(&d_x_mapped, x_mapped, 0);
    cudaHostGetDevicePointer(&d_y_mapped, y_mapped, 0);

    for (int i = 0; i < N; i++) {
        x_mapped[i] = 1.0f;
        y_mapped[i] = 2.0f;
    }

    // No explicit H->D transfer needed
    h2d_time = 0.0f;

    // Time kernel execution (transfers happen implicitly)
    cudaEventRecord(start);
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x_mapped, d_y_mapped);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    // No explicit D->H transfer needed
    d2h_time = 0.0f;

    printf("  Average H->D time: %.3f μs\n", h2d_time * 1000 / 2);
    printf("  Kernel time: %.3f μs\n", kernel_time * 1000);
    printf("  D->H time: %.3f μs\n", d2h_time * 1000);
    printf("  Total time: %.3f μs\n\n", (h2d_time + kernel_time + d2h_time) * 1000);

    cudaFreeHost(x_mapped);
    cudaFreeHost(y_mapped);

    //-----------------------------------------------------------------------//
    // Unified Memory                                                        //
    //-----------------------------------------------------------------------//
    printf("=== Unified Memory ===\n");

    float *x_unified, *y_unified;
    cudaMallocManaged(&x_unified, N * sizeof(float));
    cudaMallocManaged(&y_unified, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x_unified[i] = 1.0f;
        y_unified[i] = 2.0f;
    }

    // No explicit H->D transfer needed
    h2d_time = 0.0f;

    // Time kernel execution (migrations happen implicitly)
    cudaEventRecord(start);
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, x_unified, y_unified);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    // No explicit D->H transfer needed
    d2h_time = 0.0f;

    printf("  Average H->D time: %.3f μs\n", h2d_time * 1000 / 2);
    printf("  Kernel time: %.3f μs\n", kernel_time * 1000);
    printf("  D->H time: %.3f μs\n", d2h_time * 1000);
    printf("  Total time: %.3f μs\n\n", (h2d_time + kernel_time + d2h_time) * 1000);

    cudaFree(x_unified);
    cudaFree(y_unified);

    //-----------------------------------------------------------------------//
    // Cleanup                                                               //
    //-----------------------------------------------------------------------//
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}