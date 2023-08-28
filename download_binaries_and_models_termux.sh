#!$PREFIX/bin/bash

set -ue

pkg install -y aria2

mkdir -p rwkv
cd rwkv

ARG=${1:-2}

if (( $ARG == 0 )); then
    echo "Downloading binaries only"
elif (( $ARG == 1 )); then
    echo "Downloading binaries and 0.1b models"
elif (( $ARG == 2 )); then
    echo "Downloading binaries and 1.5b and 0.1b models"
else
    echo "Invalid argument"
    exit 1
fi
sleep 1
aria2c -c -x16 -s16 https://github.com/daquexian/faster-rwkv/releases/download/v0.0.1/faster-rwkv-android.zip -o faster-rwkv-android.zip
unzip -o faster-rwkv-android.zip
rm faster-rwkv-android.zip
aria2c -c -x16 -s16 https://github.com/daquexian/faster-rwkv/raw/v0.0.1/tokenizer_model -o tokenizer_model

if (( $ARG >= 1 )); then
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/main/rwkv-4-0.1b.bin -o rwkv-4-0.1b.bin
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/main/rwkv-4-0.1b.param -o rwkv-4-0.1b.param
    echo 'LD_LIBRARY_PATH=`pwd` ./chat tokenizer_model rwkv-4-0.1b "ncnn fp16"' > run_0.1b.sh
    chmod +x run_0.1b.sh
fi
if (( $ARG >= 2 )); then
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/main/rwkv-4-chntuned-1.5b.bin -o rwkv-4-chntuned-1.5b.bin
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/main/rwkv-4-chntuned-1.5b.param -o rwkv-4-chntuned-1.5b.param
    echo 'LD_LIBRARY_PATH=`pwd` ./chat tokenizer_model rwkv-4-chntuned-1.5b "ncnn fp16"' > run_1.5b.sh
    chmod +x run_1.5b.sh
fi

if (( $ARG == 0 )); then
    echo "Done! The binaries are inside directory 'rwkv'"
elif (( $ARG == 1 )); then
    echo "Done! Now you can run the chatbot by entering into directory 'rwkv' and running './run_0.1b.sh'"
elif (( $ARG == 2 )); then
    echo "Done! Now you can run the chatbot by entering into directory 'rwkv' and running './run_1.5b.sh' or './run_0.1b.sh'"
fi
