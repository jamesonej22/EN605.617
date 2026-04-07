inline uint rotl(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

inline void quarter_round(uint* a, uint* b, uint* c, uint* d) {
    *a += *b;
    *d ^= *a;
    *d = rotl(*d, 16);
    *c += *d;
    *b ^= *c;
    *b = rotl(*b, 12);
    *a += *b;
    *d ^= *a;
    *d = rotl(*d, 8);
    *c += *d;
    *b ^= *c;
    *b = rotl(*b, 7);
}

inline void initialize_state(uint* state, const uchar* key, const uchar* nonce, const uint counter,
                             int gid) {
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;

    // Key
    const __global uint4* key4 = (const __global uint4*)key;
    uint4 k0 = key4[0];
    uint4 k1 = key4[1];

    state[4] = k0.x;
    state[5] = k0.y;
    state[6] = k0.z;
    state[7] = k0.w;
    state[8] = k1.x;
    state[9] = k1.y;
    state[10] = k1.z;
    state[11] = k1.w;

    // Counter + nonce
    state[12] = counter + gid;

    const __global uint* nonce32 = (const __global uint*)nonce;
    state[13] = nonce32[0];
    state[14] = nonce32[1];
    state[15] = nonce32[2];
}

__kernel void chacha_kernel(__global const uchar* input, __global uchar* output,
                            __global const uchar* key, __global const uchar* nonce,
                            const uint counter, const uint input_size) {
    int gid = get_global_id(0);
    uint block_start = gid * 64;

    if (block_start >= input_size) return;
    uint state[16];
    uint working[16];
    initialize_state(state, key, nonce, counter, gid);

    for (int i = 0; i < 16; i++) {
        working[i] = state[i];
    }

    for (int i = 0; i < 10; i++) {
        quarter_round(&working[0], &working[4], &working[8], &working[12]);
        quarter_round(&working[1], &working[5], &working[9], &working[13]);
        quarter_round(&working[2], &working[6], &working[10], &working[14]);
        quarter_round(&working[3], &working[7], &working[11], &working[15]);

        quarter_round(&working[0], &working[5], &working[10], &working[15]);
        quarter_round(&working[1], &working[6], &working[11], &working[12]);
        quarter_round(&working[2], &working[7], &working[8], &working[13]);
        quarter_round(&working[3], &working[4], &working[9], &working[14]);
    }

    for (int i = 0; i < 16; i++) {
        working[i] += state[i];
    }

    __global const uint4* in4 = (const __global uint4*)(input + block_start);
    __global uint4* out4 = (__global uint4*)(output + block_start);

    uint4 r0 = (uint4)(working[0], working[1], working[2], working[3]);
    uint4 r1 = (uint4)(working[4], working[5], working[6], working[7]);
    uint4 r2 = (uint4)(working[8], working[9], working[10], working[11]);
    uint4 r3 = (uint4)(working[12], working[13], working[14], working[15]);

    out4[0] = in4[0] ^ r0;
    out4[1] = in4[1] ^ r1;
    out4[2] = in4[2] ^ r2;
    out4[3] = in4[3] ^ r3;
}