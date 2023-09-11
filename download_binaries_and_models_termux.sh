#!$PREFIX/bin/bash

set -ue

pkg install -y aria2

mkdir -p rwkv
cd rwkv

ARG=${1:-2}

if (( $ARG == 0 )); then
    echo "Downloading binaries only"
elif (( $ARG == 1 )); then
    echo "Downloading binaries and 0.1b int8 models"
elif (( $ARG == 2 )); then
    echo "Downloading binaries and 1.5b and 0.1b int8 models"
else
    echo "Invalid argument"
    exit 1
fi
sleep 1
aria2c -c -x16 -s16 https://github.com/daquexian/faster-rwkv/releases/download/v0.0.4/chat -o chat
aria2c -c -x16 -s16 https://github.com/daquexian/faster-rwkv/raw/v0.0.4/tokenizer_model -o tokenizer_model

if (( $ARG >= 1 )); then
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.bin -o RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.bin
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.param -o RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.param
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.config -o RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.config
    echo 'FR_SHOW_SPEED=1 ./chat tokenizer_model RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8 "ncnn auto"' > run_0.1b_int8.sh
    chmod +x run_0.1b_int8.sh
fi
if (( $ARG >= 2 )); then
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.bin -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.bin
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.param -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.param
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.config -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.config
    echo 'FR_SHOW_SPEED=1 ./chat tokenizer_model RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8 "ncnn auto"' > run_1.5b_int8.sh
    chmod +x run_1.5b_int8.sh
fi

if (( $ARG == 0 )); then
    echo "Done! The binaries are inside directory 'rwkv'"
elif (( $ARG == 1 )); then
    echo "Done! Now you can run the chatbot by entering into directory 'rwkv' and running './run_0.1b_int8.sh'"
elif (( $ARG == 2 )); then
    echo "Done! Now you can run the chatbot by entering into directory 'rwkv' and running './run_1.5b_int8.sh' or './run_0.1b_int8.sh'"
fi
