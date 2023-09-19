#!/usr/bin/env bash

if [[ -z $ANDROID_HOME || -z $ANDROID_NDK ]]; then
    echo "Please set ANDROID_HOME and ANDROID_NDK environment variables first"
    exit 1
fi

set -uex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../build-android-aar
cd $SCRIPT_DIR/../build-android-aar
cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DANDROID_NDK=$ANDROID_NDK -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -GNinja -DFR_ENABLE_NCNN=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release ..
ninja
mkdir -p $SCRIPT_DIR/java/faster-rwkv-java/src/main/jniLibs/arm64-v8a/
cp aar/libfaster_rwkv_jni.so $SCRIPT_DIR/java/faster-rwkv-java/src/main/jniLibs/arm64-v8a/

cd $SCRIPT_DIR/java
./gradlew clean build

echo "Done! The AAR file is in $SCRIPT_DIR/java/faster-rwkv-java/build/outputs/aar/"
