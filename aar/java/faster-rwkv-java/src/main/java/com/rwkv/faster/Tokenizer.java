package com.rwkv.faster;

public class Tokenizer {
    public Tokenizer(String path) {
        init(path);
    }
    public String decode(int id) {
        return decodeSingle(id);
    }
    public String decode(int[] ids) {
        return decodeSeq(ids);
    }
    public native int[] encode(String str);

    private native void init(String path);

    private native String decodeSingle(int id);
    private native String decodeSeq(int[] ids);

    private long nativeHandle;
}

