package com.rwkv.faster;

public class Sampler {
    public Sampler() {
        init();
    }
    public native int sample(float[] logits, float temperature, int top_k, float top_p);

    private native void init();
    private long nativeHandle;
}
