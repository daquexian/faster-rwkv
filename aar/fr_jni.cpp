#include <memory>
#include <jni.h>
#include <string>
#include <android/log.h>

#include "jni_handle.h"
#include <pipeline.h>

#include <android/log.h>

#define LOG_TAG "RWKV Demo"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

jint throwException(JNIEnv *env, std::string message);

using rwkv::Model;
using rwkv::Pipeline;
using rwkv::Tokenizer;

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

extern "C" JNIEXPORT void JNICALL Java_com_rwkv_faster_Pipeline_init(
    JNIEnv *env, jobject obj /* this */, jstring jModelPath, jstring jTokenizerPath, jstring jStrategy) {

  std::string model_path(to_cpp_string(env, jModelPath));
  std::string tokenizer_path(to_cpp_string(env, jTokenizerPath));
  std::string strategy(to_cpp_string(env, jStrategy));

  auto model = std::make_shared<Model>(model_path, strategy);
  auto tokenizer = std::make_shared<Tokenizer>(tokenizer_path);
  auto pipeline = new std::shared_ptr<Pipeline>(new Pipeline(model, tokenizer));
  setHandle(env, obj, pipeline);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_rwkv_faster_Pipeline_Run(
    JNIEnv *env, jobject obj /* this */, jstring jInput, jobject callback) {
  auto pipeline = getHandle<Pipeline>(env, obj);
  std::string input(to_cpp_string(env, jInput));
  auto output = pipeline->Run(input);
  return env->NewStringUTF(output.c_str());
}
