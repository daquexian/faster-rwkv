package com.rwkv.faster;

import android.content.res.AssetManager;

public class Tokenizer {
    public Tokenizer(String path) {
        init(path);
    }
    public Tokenizer(String path, AssetManager mgr) {
        initWithAssetManager(path, mgr);
    }
    public String decode(int id) {
        return decodeSingle(id);
    }
    public String decode(int[] ids) {
        return decodeSeq(ids);
    }
    public native int[] encode(String str);

    private native void init(String path);
    private native void initWithAssetManager(String path, AssetManager mgr);

    private native String decodeSingle(int id);
    private native String decodeSeq(int[] ids);

    private long nativeHandle;
}

