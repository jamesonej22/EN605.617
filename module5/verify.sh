#!/usr/bin/env bash

HEADER_SIZE=17
KEY=2b7e151628aed2a6abf7158809cf4f3c   
FILES=(
    encrypted_constant.ppm
    encrypted___global.ppm
    encrypted_shared_g.ppm
    encrypted_shared_c.ppm
)

for f in "${FILES[@]}"; do
    echo -n "Comparing encrypted_host.ppm with $f ... "
    if cmp -s encrypted_host.ppm "$f"; then
        echo "matches"
    else
        echo "difference found!"
    fi
done

echo
echo "Extracting pixel data for encryption with OpenSSL ..."

# Extract original pixel data (from host output)
dd if=jhu.ppm of=pixels.bin bs=1 skip=$HEADER_SIZE

# Encrypt pixel data with OpenSSL and report timing
echo
echo -n "Running OpenSSL encryption ... "

START=$(date +%s%N)

openssl enc -aes-128-ecb -K $KEY -nosalt -nopad \
    -in pixels.bin -out openssl_encrypted.bin

END=$(date +%s%N)

ELAPSED_NS=$((END - START))
ELAPSED_MS=$((ELAPSED_NS / 1000000))

echo "time taken: ${ELAPSED_MS} ms"

echo
echo "Extracting pixel data of encrypted image for comparison ... "

# Compare pixel data only
dd if=encrypted_host.ppm of=encrypted_pixels.bin bs=1 skip=$HEADER_SIZE

echo
echo -n "Comparing OpenSSL result with encrypted_host.ppm pixel data ... "
if cmp -s encrypted_pixels.bin openssl_encrypted.bin; then
    echo "matches"
else
    echo "difference found!"
fi

# Cleanup
rm -f pixels.bin openssl_encrypted.bin encrypted_pixels.bin

echo
echo "Done."