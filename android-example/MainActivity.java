// NOTE: Only for reference

package com.rwkv.demo;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.ColorSpace;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;
import android.widget.TextView;
import com.rwkv.faster.*;

import com.rwkv.demo.databinding.ActivityMainBinding;

import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'faster_rwkv' library on application startup.
    static {
        System.loadLibrary("faster_rwkv_jni");
    }

    private ActivityMainBinding binding;
    private Model model;
    private ABCTokenizer tokenizer;
    private Sampler sampler;

    private static int[] add2BeginningOfArray(int[] elements, int element)
    {
        int[] newArray = Arrays.copyOf(elements, elements.length + 1);
        newArray[0] = element;
        System.arraycopy(elements, 0, newArray, 1, elements.length);

        return newArray;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // IMPORTANT! Request for permission to access all files.
        // This is required for the app to access the model file. Otherwise, the app will crash.
        // manifest.xml should be also updated, declaring `MANAGE_EXTERNAL_STORAGE` permission.
        // See https://developer.android.com/training/data-storage/manage-all-files
        // These seem only required for Android 11 and above. But I don't check the version here.
        // What's more, the following piece of code is copied from stackoverflow.
        if (Environment.isExternalStorageManager()) {
            //todo when permission is granted
        } else {
            //request for the permission
            Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
            Uri uri = Uri.fromParts("package", getPackageName(), null);
            intent.setData(uri);
            startActivity(intent);
        }

        String input = "S:2";
        String result = input;

        model = new Model("/sdcard/Download/rwkv/RWKV-4-ABC-82M-v1-20230805-ctx1024-ncnn", "ncnn fp32");
        tokenizer = new ABCTokenizer();
        sampler = new Sampler();

        int BOS_ID = 2;
        int EOS_ID = 3;
        float temperature = 1;
        int top_k = 1;
        float top_p = 0.0f;

        int[] input_ids = tokenizer.encode(input);
        input_ids = add2BeginningOfArray(input_ids, BOS_ID);
        float[] logits = model.run(input_ids);
        for (int i = 0; i < 1024; i++) {
            int output_id = sampler.sample(logits, temperature, top_k, top_p);
            if (output_id == EOS_ID) {
                break;
            }
            String output = tokenizer.decode(output_id);
            result += output;
            logits = model.run(output_id);
        }
        Log.i("xxx", result);
    }
}
