/**
 * @file aes_cpu.cc
 * @author Eric Jameson
 * @brief Implementation of CPU-only functions used in Advanced Encryption Standard (AES) for
 * comparison with GPU kernels. See https://en.wikipedia.org/wiki/Advanced_Encryption_Standard for
 * detailed descriptions of the algorithms and relevant pseudocode.
 */

#include "aes_cpu.hh"

#include <omp.h>

#include <cstdint>

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

void aes_sub_bytes_cpu(uint8_t* state, const uint8_t* sbox) {
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = sbox[state[i]];
    }
}

void aes_shift_rows_cpu(uint8_t* state) {
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

uint8_t aes_galois_multiplication_cpu(uint8_t a, uint8_t b) {
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

void aes_mix_columns_cpu(uint8_t* state) {
    for (int i = 0; i < STATE_DIMENSION; i++) {
        int start_idx = i * STATE_DIMENSION;
        uint8_t s0 = state[start_idx];
        uint8_t s1 = state[start_idx + 1];
        uint8_t s2 = state[start_idx + 2];
        uint8_t s3 = state[start_idx + 3];

        state[start_idx] = aes_galois_multiplication_cpu(0x02, s0) ^
                           aes_galois_multiplication_cpu(0x03, s1) ^ s2 ^ s3;
        state[start_idx + 1] = s0 ^ aes_galois_multiplication_cpu(0x02, s1) ^
                               aes_galois_multiplication_cpu(0x03, s2) ^ s3;
        state[start_idx + 2] = s0 ^ s1 ^ aes_galois_multiplication_cpu(0x02, s2) ^
                               aes_galois_multiplication_cpu(0x03, s3);
        state[start_idx + 3] = aes_galois_multiplication_cpu(0x03, s0) ^ s1 ^ s2 ^
                               aes_galois_multiplication_cpu(0x02, s3);
    }
}

void aes_add_round_key_cpu(uint8_t* state, const uint8_t* round_keys, int round) {
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] ^= round_keys[round * KEY_SIZE + i];
    }
}

void aes_encrypt_cpu(const uint8_t* input, uint8_t* output, const uint8_t* round_keys,
                     int blocks_to_encrypt) {
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < blocks_to_encrypt; idx++) {
        uint8_t state[STATE_SIZE];
        for (int i = 0; i < STATE_SIZE; i++) {
            state[i] = input[idx * STATE_SIZE + i];
        }

        aes_add_round_key_cpu(state, round_keys, 0);

        for (int round = 1; round < NUM_ROUNDS; round++) {
            aes_sub_bytes_cpu(state, sbox);
            aes_shift_rows_cpu(state);
            aes_mix_columns_cpu(state);
            aes_add_round_key_cpu(state, round_keys, round);
        }

        aes_sub_bytes_cpu(state, sbox);
        aes_shift_rows_cpu(state);
        aes_add_round_key_cpu(state, round_keys, NUM_ROUNDS);

        for (int i = 0; i < STATE_SIZE; i++) {
            output[idx * STATE_SIZE + i] = state[i];
        }
    }
}