#!/usr/bin/env bash

set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR
mkdir -p build-host-protoc
cd build-host-protoc
cmake -DFR_BUILD_PROTOBUF=ON -DFR_ENABLE_NCNN=OFF -GNinja ..
ninja protoc
