find_package(JNI REQUIRED)

add_library(faster_rwkv_jni
    SHARED
    fr_jni.cpp
    jni_handle.h
    )

find_library(
    log-lib
    log
    )

target_link_libraries(faster_rwkv_jni
    faster_rwkv
    JNI::JNI
    ${log-lib}
    )
