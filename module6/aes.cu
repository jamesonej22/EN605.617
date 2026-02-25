/**
 * @file aes.cu
 * @author Eric Jameson
 * @brief Implementation of functions used in the Advanced Encryption Standard (AES). All functions
 * are described on Wikipedia: https://en.wikipedia.org/wiki/Advanced_Encryption_Standard
 */

#include <cstdint>

#include "aes.cuh"

__constant__ uint8_t device_sbox[SBOX_SIZE];
__constant__ uint8_t device_round_keys[TOTAL_KEY_SIZE];

void aes_expand_key(uint8_t* round_keys, const uint8_t* key) {
    int current_key_size = KEY_SIZE;
    uint8_t temp_word[WORD_SIZE];

    for (int i = 0; i < KEY_SIZE; i++) {
        round_keys[i] = key[i];
    }

    while (current_key_size < TOTAL_KEY_SIZE) {
        for (int i = 0; i < WORD_SIZE; i++) {
            temp_word[i] = round_keys[current_key_size - WORD_SIZE + i];
        }

        if (current_key_size % KEY_SIZE == 0) {
            uint8_t temp = temp_word[0];
            temp_word[0] = temp_word[1];
            temp_word[1] = temp_word[2];
            temp_word[2] = temp_word[3];
            temp_word[3] = temp;

            for (int i = 0; i < WORD_SIZE; i++) {
                temp_word[i] = sbox[temp_word[i]];
            }

            temp_word[0] ^= round_constants[current_key_size / KEY_SIZE - 1];
        }

        for (int i = 0; i < WORD_SIZE; i++) {
            round_keys[current_key_size] = round_keys[current_key_size - KEY_SIZE] ^ temp_word[i];
            current_key_size++;
        }
    }
}

__device__ void aes_load_shared_memory(uint8_t* shared_sbox, uint8_t* shared_round_keys) {
    for (int i = threadIdx.x; i < SBOX_SIZE; i += blockDim.x) {
        shared_sbox[i] = device_sbox[i];
    }

    for (int i = threadIdx.x; i < TOTAL_KEY_SIZE; i += blockDim.x) {
        shared_round_keys[i] = device_round_keys[i];
    }
}

__device__ void aes_sub_bytes(uint8_t* state, const uint8_t* sbox) {
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = sbox[state[i]];
    }
}

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

__device__ void aes_add_round_key(uint8_t* state, const uint8_t* round_keys, int round) {
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] ^= round_keys[round * KEY_SIZE + i];
    }
}

__device__ void aes_encrypt(const uint8_t* input, uint8_t* output, const uint8_t* sbox,
                            const uint8_t* round_keys, size_t thread_idx) {
    uint8_t state[STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = input[thread_idx * STATE_SIZE + i];
    }

    aes_add_round_key(state, round_keys, 0);

    for (int round = 1; round < NUM_ROUNDS; round++) {
        aes_sub_bytes(state, sbox);
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, round_keys, round);
    }

    aes_sub_bytes(state, sbox);
    aes_shift_rows(state);
    aes_add_round_key(state, round_keys, NUM_ROUNDS);

    for (int i = 0; i < STATE_SIZE; i++) {
        output[thread_idx * STATE_SIZE + i] = state[i];
    }
}

__device__ void aes_encrypt_ctr(const uint8_t* input, uint8_t* output, const uint8_t* sbox,
                                const uint8_t* round_keys, uint64_t nonce, uint64_t counter) {
    uint8_t state[STATE_SIZE];

    // Construct counter block
    for (int i = 0; i < 8; i++) {
        state[i] = (nonce >> (56 - 8 * i)) & 0xff;
        state[8 + i] = (counter >> (56 - 8 * i)) & 0xff;
    }

    aes_add_round_key(state, round_keys, 0);

    for (int round = 1; round < NUM_ROUNDS; round++) {
        aes_sub_bytes(state, sbox);
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, round_keys, round);
    }

    aes_sub_bytes(state, sbox);
    aes_shift_rows(state);
    aes_add_round_key(state, round_keys, NUM_ROUNDS);

    for (int i = 0; i < STATE_SIZE; i++) {
        output[i] = input[i] ^ state[i];
    }
}

__global__ void aes_encrypt_ecb_kernel(const uint8_t* input, uint8_t* output, size_t input_size,
                                       size_t /*unused*/) {
    __shared__ uint8_t shared_sbox[SBOX_SIZE];
    __shared__ uint8_t shared_round_keys[TOTAL_KEY_SIZE];

    aes_load_shared_memory(shared_sbox, shared_round_keys);
    __syncthreads();

    size_t grid_size = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blocks_to_encrypt = input_size / STATE_SIZE;

    for (; idx < blocks_to_encrypt; idx += grid_size) {
        aes_encrypt(input, output, shared_sbox, shared_round_keys, idx);
    }
}

__global__ void aes_encrypt_ctr_kernel(const uint8_t* input, uint8_t* output, size_t input_size,
                                       size_t ctr_start) {
    __shared__ uint8_t shared_sbox[SBOX_SIZE];
    __shared__ uint8_t shared_round_keys[TOTAL_KEY_SIZE];

    aes_load_shared_memory(shared_sbox, shared_round_keys);
    __syncthreads();

    size_t grid_size = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blocks_to_encrypt = input_size / STATE_SIZE;
    uint64_t ctr_nonce = nonce;

    for (; idx < blocks_to_encrypt; idx += grid_size) {
        uint64_t counter = idx + ctr_start;
        aes_encrypt_ctr(input + idx * STATE_SIZE, output + idx * STATE_SIZE, shared_sbox,
                        shared_round_keys, ctr_nonce, counter);
    }
}