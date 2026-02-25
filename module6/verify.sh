#!/usr/bin/env bash

HEADER_SIZE=17
KEY=2b7e151628aed2a6abf7158809cf4f3c   
NONCE=123456789abcdef00000000000000000
STREAM_COUNTS=(1 2 4 8 16)

echo
echo "Comparing multi-stream output with single stream for both modes ..."

for N in "${STREAM_COUNTS[@]}"; do
    [[ $N -eq 1 ]] && continue

    for MODE in ecb ctr; do
        REF="encrypted_${MODE}_1.ppm"
        CANDIDATE="encrypted_${MODE}_${N}.ppm"

        echo -n "encrypted_${MODE}_1.ppm vs ${CANDIDATE} ... "
        if cmp -s "$REF" "$CANDIDATE"; then
            echo "matches"
        else
            echo "difference found!"
        fi
    done
done


echo
echo "Extracting pixel data for encryption with OpenSSL ..."

# Extract original pixel data
tail -c +$((HEADER_SIZE + 1)) cuda.ppm > pixels.bin

# Encrypt pixel data with OpenSSL and report timing
echo -n "Running OpenSSL ECB encryption ... "

START=$(date +%s%N)

openssl enc -aes-128-ecb -K $KEY -nosalt -nopad \
    -in pixels.bin -out openssl_ecb.bin

END=$(date +%s%N)

ELAPSED_NS=$((END - START))
ELAPSED_MS=$((ELAPSED_NS / 1000000))

echo "time taken: ${ELAPSED_MS} ms"

echo -n "Running OpenSSL CTR encryption ... "

START=$(date +%s%N)

openssl enc -aes-128-ctr -K $KEY -nosalt -nopad \
    -iv $NONCE -in pixels.bin -out openssl_ctr.bin

END=$(date +%s%N)

ELAPSED_NS=$((END - START))
ELAPSED_MS=$((ELAPSED_NS / 1000000))

echo "time taken: ${ELAPSED_MS} ms"

echo "Extracting pixel data of encrypted images for comparison ... "

# Extract pixel data from encrypted image
tail -c +$((HEADER_SIZE + 1)) encrypted_ecb_1.ppm > ecb_pixels.bin
tail -c +$((HEADER_SIZE + 1)) encrypted_ctr_1.ppm > ctr_pixels.bin


echo -n "Comparing OpenSSL ECB result with encrypted_ecb_1.ppm pixel data ... "
if cmp -s ecb_pixels.bin openssl_ecb.bin; then
    echo "matches"
else
    echo "difference found!"
fi

echo -n "Comparing OpenSSL CTR result with encrypted_ctr_1.ppm pixel data ... "
if cmp -s ctr_pixels.bin openssl_ctr.bin; then
    echo "matches"
else
    echo "difference found!"
fi

# Cleanup
rm -f pixels.bin openssl_ecb.bin ecb_pixels.bin openssl_ctr.bin ctr_pixels.bin

echo
echo "Done."