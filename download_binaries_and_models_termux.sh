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
    echo "Downloading binaries, 0.1b int8 model and 1.5b int4 model"
elif (( $ARG == 2 )); then
    echo "Downloading binaries, 0.1b int8 model, 1.5b int4 and int8 model"
else
    echo "Invalid argument"
    exit 1
fi
sleep 1
rm -f chat
aria2c -c -x16 -s16 https://github.com/daquexian/faster-rwkv/releases/download/v0.0.14/chat-android -o chat
chmod +x chat
aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/02382cd/tokenizers/world_tokenizer -o world_tokenizer

if (( $ARG >= 1 )); then
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.bin -o RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.bin
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.param -o RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.param
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.config -o RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8.config
    echo 'SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ); FR_SHOW_SPEED=1 $SCRIPT_DIR/chat $SCRIPT_DIR/world_tokenizer $SCRIPT_DIR/RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn-int8 "ncnn auto"' > run_0.1b_int8.sh
    chmod +x run_0.1b_int8.sh
fi
if (( $ARG >= 2 )); then
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/efc05d6/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096/ncnn/int4/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int4.param -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int4.param
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/efc05d6/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096/ncnn/int4/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int4.config -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int4.config
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/efc05d6/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096/ncnn/int4/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int4.bin -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int4.bin
    echo 'SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ); FR_SHOW_SPEED=1 $SCRIPT_DIR/chat $SCRIPT_DIR/world_tokenizer $SCRIPT_DIR/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int4 "ncnn auto"' > run_1.5b_int4.sh
    chmod +x run_1.5b_int4.sh
fi
if (( $ARG >= 3 )); then
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.bin -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.bin
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.param -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.param
    aria2c -c -x16 -s16 https://huggingface.co/daquexian/fr-models/resolve/0549757/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.config -o RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8.config
    echo 'SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ); FR_SHOW_SPEED=1 $SCRIPT_DIR/chat $SCRIPT_DIR/world_tokenizer $SCRIPT_DIR/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-ncnn-int8 "ncnn auto"' > run_1.5b_int8.sh
    chmod +x run_1.5b_int8.sh
fi

if (( $ARG == 0 )); then
    echo "Done! The binaries are inside directory 'rwkv'"
elif (( $ARG == 1 )); then
    echo "Done! Now you can run the chatbot by running './rwkv/run_0.1b_int8.sh'"
elif (( $ARG == 2 )); then
    echo "Done! Now you can run the chatbot by running './rwkv/run_1.5b_int4.sh' or './rwkv/run_0.1b_int8.sh'"
elif (( $ARG == 3 )); then
    echo "Done! Now you can run the chatbot by running './rwkv/run_1.5b_int4.sh',  './rwkv/run_1.5b_int8.sh' or './rwkv/run_0.1b_int8.sh'"
fi
