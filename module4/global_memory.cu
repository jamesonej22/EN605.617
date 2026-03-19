/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static const int WORK_SIZE = 256;

#define NUM_ELEMENTS (1u << 20)

typedef struct {
    unsigned int a;
    unsigned int b;
    unsigned int c;
    unsigned int d;
} INTERLEAVED_T;

typedef struct {
    unsigned int a[NUM_ELEMENTS];
    unsigned int b[NUM_ELEMENTS];
    unsigned int c[NUM_ELEMENTS];
    unsigned int d[NUM_ELEMENTS];
} NON_INTERLEAVED_T;

__host__ cudaEvent_t get_time(void) {
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
    return time;
}

__host__ float add_test_non_interleaved_cpu(NON_INTERLEAVED_T* host_dest_ptr,
                                            NON_INTERLEAVED_T* const host_src_ptr,
                                            const unsigned int iter,
                                            const unsigned int num_elements) {
    cudaEvent_t start_time = get_time();

    for (unsigned int tid = 0; tid < num_elements; tid++) {
        for (unsigned int i = 0; i < iter; i++) {
            host_dest_ptr->a[tid] += host_src_ptr->a[tid];
            host_dest_ptr->b[tid] += host_src_ptr->b[tid];
            host_dest_ptr->c[tid] += host_src_ptr->c[tid];
            host_dest_ptr->d[tid] += host_src_ptr->d[tid];
        }
    }

    cudaEvent_t end_time = get_time();
    cudaEventSynchronize(end_time);

    float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
    return delta;
}

__host__ float add_test_interleaved_cpu(INTERLEAVED_T* const host_dest_ptr,
                                        const INTERLEAVED_T* const host_src_ptr,
                                        const unsigned int iter, const unsigned int num_elements) {
    cudaEvent_t start_time = get_time();
    for (unsigned int tid = 0; tid < num_elements; tid++) {
        // printf("tid: %u ", tid);
        for (unsigned int i = 0; i < iter; i++) {
            // printf("iteration: %u\n", iter);
            host_dest_ptr[tid].a += host_src_ptr[tid].a;
            host_dest_ptr[tid].b += host_src_ptr[tid].b;
            host_dest_ptr[tid].c += host_src_ptr[tid].c;
            host_dest_ptr[tid].d += host_src_ptr[tid].d;
        }
    }

    cudaEvent_t end_time = get_time();
    cudaEventSynchronize(end_time);

    float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
    return delta;
}

__global__ void add_kernel_interleaved(INTERLEAVED_T* const dest_ptr,
                                       const INTERLEAVED_T* const src_ptr, const unsigned int iter,
                                       const unsigned int num_elements) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < num_elements) {
        for (unsigned int i = 0; i < iter; i++) {
            dest_ptr[tid].a += src_ptr[tid].a;
            dest_ptr[tid].b += src_ptr[tid].b;
            dest_ptr[tid].c += src_ptr[tid].c;
            dest_ptr[tid].d += src_ptr[tid].d;
        }
    }
}

__global__ void add_kernel_non_interleaved(NON_INTERLEAVED_T* const dest_ptr,
                                           NON_INTERLEAVED_T* const src_ptr,
                                           const unsigned int iter,
                                           const unsigned int num_elements) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_elements) return;
    for (unsigned int i = 0; i < iter; i++) {
        dest_ptr->a[tid] += src_ptr->a[tid];
        dest_ptr->b[tid] += src_ptr->b[tid];
        dest_ptr->c[tid] += src_ptr->c[tid];
        dest_ptr->d[tid] += src_ptr->d[tid];
    }
}

__host__ float add_test_interleavedGPU(INTERLEAVED_T* const host_dest_ptr,
                                       const INTERLEAVED_T* const host_src_ptr,
                                       const unsigned int iter, const unsigned int num_elements) {
    const unsigned int num_threads = 256;
    const unsigned int num_blocks = (num_elements + (num_threads - 1)) / num_threads;

    const size_t num_bytes = (sizeof(INTERLEAVED_T) * num_elements);
    INTERLEAVED_T* device_dest_ptr;
    INTERLEAVED_T* device_src_ptr;

    cudaMalloc((void**)&device_src_ptr, num_bytes);
    cudaMalloc((void**)&device_dest_ptr, num_bytes);

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start, 0);
    cudaEventCreate(&kernel_stop, 0);

    cudaStream_t test_stream;
    cudaStreamCreate(&test_stream);

    cudaMemcpy(device_src_ptr, host_src_ptr, num_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(kernel_start, 0);

    add_kernel_interleaved<<<num_blocks, num_threads>>>(device_dest_ptr, device_src_ptr, iter,
                                                        num_elements);

    cudaEventRecord(kernel_stop, 0);

    cudaEventSynchronize(kernel_stop);

    float delta = 0.0F;
    cudaEventElapsedTime(&delta, kernel_start, kernel_stop);

    cudaFree(device_src_ptr);
    cudaFree(device_dest_ptr);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaStreamDestroy(test_stream);

    return delta;
}

__host__ float add_test_non_interleavedGPU(NON_INTERLEAVED_T* const host_dest_ptr,
                                           const NON_INTERLEAVED_T* const host_src_ptr,
                                           const unsigned int iter,
                                           const unsigned int num_elements) {
    const unsigned int num_threads = 256;
    const unsigned int num_blocks = (num_elements + (num_threads - 1)) / num_threads;

    const size_t num_bytes = (sizeof(NON_INTERLEAVED_T));
    NON_INTERLEAVED_T* device_dest_ptr;
    NON_INTERLEAVED_T* device_src_ptr;

    cudaMalloc((void**)&device_src_ptr, num_bytes);
    cudaMalloc((void**)&device_dest_ptr, num_bytes);

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start, 0);
    cudaEventCreate(&kernel_stop, 0);

    cudaStream_t test_stream;
    cudaStreamCreate(&test_stream);

    cudaMemcpy(device_src_ptr, host_src_ptr, num_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(kernel_start, 0);

    add_kernel_non_interleaved<<<num_blocks, num_threads>>>(device_dest_ptr, device_src_ptr, iter,
                                                            num_elements);
    cudaEventRecord(kernel_stop, 0);

    cudaEventSynchronize(kernel_stop);

    float delta = 0.0F;
    cudaEventElapsedTime(&delta, kernel_start, kernel_stop);

    cudaFree(device_src_ptr);
    cudaFree(device_dest_ptr);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaStreamDestroy(test_stream);

    return delta;
}

__host__ float add_test_non_interleaved(NON_INTERLEAVED_T host_dest_ptr,
                                        const NON_INTERLEAVED_T host_src_ptr,
                                        const unsigned int iter, const unsigned int num_elements) {
    const unsigned int num_threads = 256;
    const unsigned int num_blocks = (num_elements + (num_threads - 1)) / num_threads;

    const size_t num_bytes = (sizeof(INTERLEAVED_T) * num_elements);
    NON_INTERLEAVED_T* device_dest_ptr;
    NON_INTERLEAVED_T* device_src_ptr;

    cudaMalloc((void**)&device_src_ptr, num_bytes);
    cudaMalloc((void**)&device_dest_ptr, num_bytes);

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start, 0);
    cudaEventCreate(&kernel_stop, 0);

    cudaMemcpy(device_src_ptr, &host_src_ptr, num_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(kernel_start, 0);
    add_kernel_non_interleaved<<<num_blocks, num_threads>>>(device_dest_ptr, device_src_ptr, iter,
                                                            num_elements);
    cudaEventRecord(kernel_stop, 0);
    cudaEventSynchronize(kernel_stop);

    float delta = 0.0F;
    cudaEventElapsedTime(&delta, kernel_start, kernel_stop);

    cudaFree(device_src_ptr);
    cudaFree(device_dest_ptr);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);

    return delta;
}

void execute_host_functions() {
    INTERLEAVED_T host_dest_ptr[NUM_ELEMENTS];
    INTERLEAVED_T host_src_ptr[NUM_ELEMENTS];

    for (unsigned int i = 0; i < NUM_ELEMENTS; i++) {
        host_dest_ptr[i] = INTERLEAVED_T{.a = i, .b = i, .c = i, .d = i};
        host_src_ptr[i] = INTERLEAVED_T{.a = i, .b = i, .c = i, .d = i};
    }

    float duration = add_test_interleaved_cpu(host_dest_ptr, host_src_ptr, 4, NUM_ELEMENTS);
    printf("CPU INTERLEAVED: duration: %fms\n", duration);

    NON_INTERLEAVED_T nhost_dest_ptr;
    NON_INTERLEAVED_T nhost_src_ptr;
    for (unsigned int i = 0; i < NUM_ELEMENTS; i++) {
        nhost_dest_ptr.a[i] = i;
        nhost_dest_ptr.b[i] = i;
        nhost_dest_ptr.c[i] = i;
        nhost_dest_ptr.d[i] = i;

        nhost_src_ptr.a[i] = i;
        nhost_src_ptr.b[i] = i;
        nhost_src_ptr.c[i] = i;
        nhost_src_ptr.d[i] = i;
    }

    duration = add_test_non_interleaved_cpu(nhost_dest_ptr, nhost_src_ptr, 4, NUM_ELEMENTS);
    printf("CPU NON-INTERLEAVED: duration: %fms\n", duration);
}

void init_interleaved(INTERLEAVED_T* p, unsigned int value) {
    for (unsigned int i = 0; i < NUM_ELEMENTS; i++) {
        p[i].a = value;
        p[i].b = value;
        p[i].c = value;
        p[i].d = value;
    }
}

void init_non_interleaved(NON_INTERLEAVED_T* p, unsigned int value) {
    for (unsigned int i = 0; i < NUM_ELEMENTS; i++) {
        p->a[i] = value;
        p->b[i] = value;
        p->c[i] = value;
        p->d[i] = value;
    }
}

void InterleavedAndNonInterleavedCPUtest() {
    INTERLEAVED_T* host_dest_ptr = (INTERLEAVED_T*)calloc(NUM_ELEMENTS, sizeof(INTERLEAVED_T));
    if (!host_dest_ptr) {
        return;
    }

    INTERLEAVED_T* host_src_ptr = (INTERLEAVED_T*)malloc(NUM_ELEMENTS * sizeof(INTERLEAVED_T));
    if (!host_src_ptr) {
        return;
    }
    init_interleaved(host_src_ptr, 1);

    float duration = add_test_interleaved_cpu(host_dest_ptr, host_src_ptr, 4, NUM_ELEMENTS);
    printf("interleaved on CPU duration: %fms\n", duration);

    NON_INTERLEAVED_T* nhost_dest_ptr = (NON_INTERLEAVED_T*)malloc(sizeof(NON_INTERLEAVED_T));
    if (!nhost_dest_ptr) {
        return;
    }
    NON_INTERLEAVED_T* nhost_src_ptr = (NON_INTERLEAVED_T*)malloc(sizeof(NON_INTERLEAVED_T));
    if (!nhost_src_ptr) {
        return;
    }
    init_non_interleaved(nhost_src_ptr, 1);

    float nduration = add_test_non_interleaved_cpu(nhost_dest_ptr, nhost_src_ptr, 4, NUM_ELEMENTS);
    printf("non interleaved on CPU duration: %fms\n", nduration);
}

void InterleavedAndNonInterleavedGPUtest() {
    INTERLEAVED_T* host_dest_ptr = (INTERLEAVED_T*)calloc(NUM_ELEMENTS, sizeof(INTERLEAVED_T));
    if (!host_dest_ptr) {
        return;
    }

    INTERLEAVED_T* host_src_ptr = (INTERLEAVED_T*)malloc(NUM_ELEMENTS * sizeof(INTERLEAVED_T));
    if (!host_src_ptr) {
        return;
    }
    init_interleaved(host_src_ptr, 1);
    init_interleaved(host_dest_ptr, 0);
    float duration = add_test_interleavedGPU(host_dest_ptr, host_src_ptr, 4, NUM_ELEMENTS);
    printf("interleaved on GPU duration: %fms\n", duration);

    NON_INTERLEAVED_T* nhost_dest_ptr = (NON_INTERLEAVED_T*)malloc(sizeof(NON_INTERLEAVED_T));
    if (!nhost_dest_ptr) {
        return;
    }
    NON_INTERLEAVED_T* nhost_src_ptr = (NON_INTERLEAVED_T*)malloc(sizeof(NON_INTERLEAVED_T));
    if (!nhost_src_ptr) {
        return;
    }
    init_non_interleaved(nhost_src_ptr, 1);
    init_non_interleaved(nhost_dest_ptr, 0);
    float nduration = add_test_non_interleavedGPU(nhost_dest_ptr, nhost_src_ptr, 4, NUM_ELEMENTS);
    printf("non interleaved on GPU duration: %fms\n", nduration);
}

void bitreversegpu() {
    void* d = NULL;
    unsigned int idata[WORK_SIZE], odata[WORK_SIZE];
    int i;
    for (i = 0; i < WORK_SIZE; i++) idata[i] = (unsigned int)i;

    cudaMalloc((void**)&d, sizeof(int) * WORK_SIZE);

    cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);

    bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

    cudaDeviceSynchronize();  // Wait for the GPU launched work to complete
    cudaGetLastError();

    cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

    for (i = 0; i < WORK_SIZE; i++)
        printf("Input value: %u, device output: %u, host output: %u\n", idata[i], odata[i],
               bitreverse(idata[i]));

    cudaFree((void*)d);
    cudaDeviceReset();
}

void execute_host_functions() { InterleavedAndNonInterleavedCPUtest(); }

void execute_gpu_functions() {
    // bitreversegpu();
    InterleavedAndNonInterleavedGPUtest();
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
    execute_host_functions();
    execute_gpu_functions();
    return 0;
}
