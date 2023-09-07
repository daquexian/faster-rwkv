package com.rwkv.faster;

public class ABCTokenizer {
    public ABCTokenizer() {
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
