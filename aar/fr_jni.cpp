#include <memory>
#include <jni.h>
#include <string>
#include <android/log.h>

#include "jni_handle.h"
#include <model.h>
#include <tokenizer.h>
#include <sampler.h>

#include <android/log.h>

#define LOG_TAG "RWKV Demo"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

jint throwException(JNIEnv *env, std::string message);

using rwkv::Model;
using rwkv::Tokenizer;
using rwkv::WorldTokenizer;
using rwkv::ABCTokenizer;
using rwkv::Sampler;

std::string to_cpp_string(JNIEnv *env, jstring jstr) {
  const char *ptr = env->GetStringUTFChars(jstr, nullptr);
  std::string cpp_str(ptr);
  env->ReleaseStringUTFChars(jstr, ptr);
  return cpp_str;
}

extern "C" JNIEXPORT void JNICALL Java_com_rwkv_faster_Model_init(
    JNIEnv *env, jobject obj /* this */, jstring jPath, jstring jStrategy) {
  std::string path(to_cpp_string(env, jPath));
  std::string strategy(to_cpp_string(env, jStrategy));
  auto *model =
      new std::shared_ptr<Model>(new Model(path, strategy));
  setHandle(env, obj, model);
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_rwkv_faster_Model_runSingle(
    JNIEnv *env, jobject obj /* this */, jint input_id) {
  auto model = getHandle<Model>(env, obj);
  auto output = model->Run(input_id);
  jfloatArray result = env->NewFloatArray(output.numel());
  env->SetFloatArrayRegion(result, 0, output.numel(), output.data_ptr<float>());
  return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_rwkv_faster_Model_runSeq(
    JNIEnv *env, jobject obj /* this */, jintArray jInputIds) {
  auto model = getHandle<Model>(env, obj);
  jint *arr = env->GetIntArrayElements(jInputIds, nullptr);
  std::vector<int> input_ids(arr, arr + env->GetArrayLength(jInputIds));
  env->ReleaseIntArrayElements(jInputIds, arr, 0);
  auto output = model->Run(input_ids);
  jfloatArray result = env->NewFloatArray(output.numel());
  env->SetFloatArrayRegion(result, 0, output.numel(), output.data_ptr<float>());
  return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_rwkv_faster_WorldTokenizer_init(
    JNIEnv *env, jobject obj /* this */, jstring jPath) {
  std::string path(to_cpp_string(env, jPath));
  auto *tokenizer =
      new std::shared_ptr<WorldTokenizer>(new WorldTokenizer(path));
  setHandle(env, obj, tokenizer);
}

extern "C" JNIEXPORT jintArray JNICALL Java_com_rwkv_faster_WorldTokenizer_encode(
    JNIEnv *env, jobject obj /* this */, jstring jStr) {
  auto tokenizer = getHandle<WorldTokenizer>(env, obj);
  std::string str(to_cpp_string(env, jStr));
  auto output = tokenizer->encode(str);
  jintArray result = env->NewIntArray(output.size());
  env->SetIntArrayRegion(result, 0, output.size(), output.data());
  return result;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_rwkv_faster_WorldTokenizer_decodeSingle(
    JNIEnv *env, jobject obj /* this */, jint jId) {
  auto tokenizer = getHandle<WorldTokenizer>(env, obj);
  int id = static_cast<int>(jId);
  auto output = tokenizer->decode(id);
  return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_rwkv_faster_WorldTokenizer_decodeSeq(
    JNIEnv *env, jobject obj /* this */, jintArray jIds) {
  auto tokenizer = getHandle<WorldTokenizer>(env, obj);
  jint *arr = env->GetIntArrayElements(jIds, nullptr);
  std::vector<int> ids(arr, arr + env->GetArrayLength(jIds));
  env->ReleaseIntArrayElements(jIds, arr, 0);
  auto output = tokenizer->decode(ids);
  return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_rwkv_faster_ABCTokenizer_init(
    JNIEnv *env, jobject obj /* this */) {
  auto *tokenizer =
      new std::shared_ptr<ABCTokenizer>(new ABCTokenizer());
  setHandle(env, obj, tokenizer);
}

extern "C" JNIEXPORT jintArray JNICALL Java_com_rwkv_faster_ABCTokenizer_encode(
    JNIEnv *env, jobject obj /* this */, jstring jStr) {
  auto tokenizer = getHandle<ABCTokenizer>(env, obj);
  std::string str(to_cpp_string(env, jStr));
  auto output = tokenizer->encode(str);
  jintArray result = env->NewIntArray(output.size());
  env->SetIntArrayRegion(result, 0, output.size(), output.data());
  return result;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_rwkv_faster_ABCTokenizer_decodeSingle(
    JNIEnv *env, jobject obj /* this */, jint jId) {
  auto tokenizer = getHandle<ABCTokenizer>(env, obj);
  int id = static_cast<int>(jId);
  auto output = tokenizer->decode(id);
  return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_rwkv_faster_ABCTokenizer_decodeSeq(
    JNIEnv *env, jobject obj /* this */, jintArray jIds) {
  auto tokenizer = getHandle<ABCTokenizer>(env, obj);
  jint *arr = env->GetIntArrayElements(jIds, nullptr);
  std::vector<int> ids(arr, arr + env->GetArrayLength(jIds));
  env->ReleaseIntArrayElements(jIds, arr, 0);
  auto output = tokenizer->decode(ids);
  return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_rwkv_faster_Sampler_init(
    JNIEnv *env, jobject obj /* this */) {
  auto *sampler =
      new std::shared_ptr<Sampler>(new Sampler());
  setHandle(env, obj, sampler);
}

extern "C" JNIEXPORT jint JNICALL Java_com_rwkv_faster_Sampler_sample(
    JNIEnv *env, jobject obj /* this */, jfloatArray jProbs, jfloat temperature, jint top_k, jfloat top_p) {
  auto sampler = getHandle<Sampler>(env, obj);
  jfloat *arr = env->GetFloatArrayElements(jProbs, nullptr);
  auto output = sampler->Sample(arr, env->GetArrayLength(jProbs), temperature, top_k, top_p);
  return static_cast<jint>(output);
}
