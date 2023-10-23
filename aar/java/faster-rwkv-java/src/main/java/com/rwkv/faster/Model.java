package com.rwkv.faster;

import android.content.res.AssetManager;

public class Model {
    public Model(String path, String strategy) {
        init(path, strategy);
    }
    public Model(String path, String strategy, AssetManager mgr) {
        initWithAssetManager(path, strategy, mgr);
    }
    public float[] run(int inputId) {
        return runSingle(inputId);
    }
    public float[] run(int[] inputIds) {
        return runSeq(inputIds);
    }
    public native void loadStateFile(String path);
    public void loadStateFile(String path, AssetManager mgr) {
        loadStateFileWithAssetManager(path, mgr);
    }
    public native void saveStateFile(String path);
    public native void resetStates();
    private native void init(String path, String strategy);
    private native void initWithAssetManager(String path, String strategy, AssetManager mgr);
    private native float[] runSingle(int inputId);
    private native float[] runSeq(int[] inputIds);
    private native void loadStateFileWithAssetManager(String path, AssetManager mgr);

    private long nativeHandle;
}
