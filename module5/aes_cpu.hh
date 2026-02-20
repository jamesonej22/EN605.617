/**
 * @file aes_cpu.hh
 * @author Eric Jameson
 * @brief Declaration of constants and CPU-only functions used in the Advanced Encryption Standard
 * (AES) for comparison with GPU kernels. All constants (with the exception of our static key) and
 * functions are described on Wikipedia: https://en.wikipedia.org/wiki/Advanced_Encryption_Standard
 */

#ifndef AES_CPU_H_
#define AES_CPU_H_

#include <cstdint>

/** @brief Number of elements in the state operated on during encryption. */
#define STATE_SIZE 16
/** @brief Number of rows and columns in the state, used only for loop iteration. */
#define STATE_DIMENSION 4
/** @brief Size of the key we define for the purposes of this assignment. */
#define KEY_SIZE 16
/** @brief Number of elements in the Rijndael S-box. */
#define SBOX_SIZE 256
/** @brief Number of rounds to use in encryption. */
#define NUM_ROUNDS 10
/** @brief Constant number of bits per byte, used in Multiplication in GF(2^8). */
#define BITS_PER_BYTE 8
/** @brief Number of bytes in a 32-bit word, used in key expansion. */
#define WORD_SIZE 4
/** @brief Total size of the expanded key. */
#define TOTAL_KEY_SIZE KEY_SIZE + (KEY_SIZE * NUM_ROUNDS)

/** @brief Rijndael S-box used for byte substitution in the SubBytes step of encryption. */
const uint8_t sbox[SBOX_SIZE] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

/** @brief Round constants used in key expansion. */
const uint32_t round_constants[NUM_ROUNDS] = {0x01, 0x02, 0x04, 0x08, 0x10,
                                              0x20, 0x40, 0x80, 0x1B, 0x36};

/** @brief Constant key defined for the purposes of this assignment. */
const uint8_t key[KEY_SIZE] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
};

/**
 * @brief Perform key expansion on the provided key. For more information, see:
 * https://en.wikipedia.org/wiki/AES_key_schedule
 *
 * @param[out] round_keys Expanded key to be used for encryption.
 * @param key Provided key before expansion.
 */
void aes_expand_key(uint8_t* round_keys, const uint8_t* key);

/**
 * @brief Perform the SubBytes step of encryption, using only the CPU. For more information, see:
 * https://en.wikipedia.org/wiki/Rijndael_S-box
 *
 * @param[in,out] state The current state to operate on.
 * @param sbox Substitution table to use for SubBytes.
 */
void aes_sub_bytes_cpu(uint8_t* state, const uint8_t* sbox);

/**
 * @brief Perform the ShiftRows step of encryption, using only the CPU. For more information, see:
 * https://en.wikipedia.org/wiki/Advanced_Encryption_Standard#The_ShiftRows_step
 *
 * @param[in,out] state The current state to operate on.
 */
void aes_shift_rows_cpu(uint8_t* state);

/**
 * @brief Multiply two elements of GF(2^8). Adapted from
 * https://en.wikipedia.org/wiki/Rijndael_MixColumns#Implementation_example
 *
 * @param a The first multiplicand.
 * @param b The second multiplicand.
 * @return The product of \p a and \p b in GF(2^8).
 */
uint8_t aes_galois_multiplication_cpu(uint8_t a, uint8_t b);

/**
 * @brief Perform the MixColumns step of encryption, using only the CPU. For more information, see:
 * https://en.wikipedia.org/wiki/Rijndael_MixColumns
 *
 * @param[in,out] state The current state to operate on.
 */
void aes_mix_columns_cpu(uint8_t* state);

/**
 * @brief Perform the AddRoundKey step of encryption, using only the CPU. For more information,
 * see: https://en.wikipedia.org/wiki/Advanced_Encryption_Standard#The_AddRoundKey
 *
 * @param[in,out] state The current state to operate on.
 * @param round_keys The expanded round keys used for encryption.
 * @param round The round of encryption that we are on.
 */
void aes_add_round_key_cpu(uint8_t* state, const uint8_t* round_keys, int round);

/**
 * @brief Perform the entirety of AES encryption on the input data, using only the CPU.
 *
 * @param input Plaintext to encrypt.
 * @param[out] output Location to store the encrypted ciphertext.
 * @param round_keys Expanded keys to use during encryption.
 * @param blocks_to_encrypt Number of 16-byte blocks to encrypt. Assumed to be the size of the
 * input data divided by 16.
 */
void aes_encrypt_cpu(const uint8_t* input, uint8_t* output, const uint8_t* round_keys,
                     int blocks_to_encrypt);

#endif