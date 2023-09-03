#pragma once

#include <jni.h>
#include <memory>

inline jfieldID getHandleField(JNIEnv *env, jobject obj) {
    jclass c = env->GetObjectClass(obj);
    // J is the type signature for long:
    return env->GetFieldID(c, "nativeHandle", "J");
}

template <typename T>
std::shared_ptr<T> getHandle(JNIEnv *env, jobject obj) {
    jlong handle = env->GetLongField(obj, getHandleField(env, obj));
    return *reinterpret_cast<std::shared_ptr<T>*>(handle);
}

template<typename T>
void setHandle(JNIEnv *env, jobject obj, std::shared_ptr<T> *t) {
    jlong handle = reinterpret_cast<jlong>(t);
    env->SetLongField(obj, getHandleField(env, obj), handle);
}
