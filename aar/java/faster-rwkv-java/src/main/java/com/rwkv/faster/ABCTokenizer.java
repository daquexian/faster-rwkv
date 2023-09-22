package com.rwkv.faster;

import android.util.Log;

public class ABCTokenizer {
    public ABCTokenizer() {
        Log.i("faster-rwkv", "`ABCTokenizer` is deprecated. Use `Tokenizer` instead.");
        init();
    }

    public String decode(int id) {
        return decodeSingle(id);
    }
    public String decode(int[] ids) {
        return decodeSeq(ids);
    }
    public native int[] encode(String str);

    private native void init();
    private native String decodeSingle(int id);
    private native String decodeSeq(int[] ids);

    private long nativeHandle;
}
