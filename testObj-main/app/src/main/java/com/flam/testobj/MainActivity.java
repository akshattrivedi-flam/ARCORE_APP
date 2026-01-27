package com.flam.testobj;

import android.app.Activity;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;

import com.google.ar.core.exceptions.CameraNotAvailableException;


public class MainActivity extends Activity {
    MainSurfaceView glSurfaceView;
    com.google.ar.core.Session session;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            getWindow().getAttributes().layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        }
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_LOW_PROFILE
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_IMMERSIVE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON, WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getActionBar().hide();
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        
        glSurfaceView = new MainSurfaceView(this);
        FrameLayout frameLayout = new FrameLayout(this);
        frameLayout.addView(glSurfaceView);

        android.widget.LinearLayout controls = new android.widget.LinearLayout(this);
        controls.setOrientation(android.widget.LinearLayout.HORIZONTAL);
        controls.setPadding(20, 20, 20, 20);
        controls.setBackgroundColor(android.graphics.Color.parseColor("#40000000")); // lighter background

        createVerticalSlider(controls, "SX", 1000, 100, new SliderListener() {
            @Override
            public void onProgress(int progress) {
               float scale = 0.001f + (progress / 1000.0f);
               if(glSurfaceView.mRenderer != null) glSurfaceView.mRenderer.setScaleX(scale);
            }
        });
        createVerticalSlider(controls, "SY", 1000, 100, new SliderListener() {
            @Override
            public void onProgress(int progress) {
               float scale = 0.001f + (progress / 1000.0f);
               if(glSurfaceView.mRenderer != null) glSurfaceView.mRenderer.setScaleY(scale);
            }
        });
        createVerticalSlider(controls, "SZ", 1000, 100, new SliderListener() {
            @Override
            public void onProgress(int progress) {
               float scale = 0.001f + (progress / 1000.0f);
               if(glSurfaceView.mRenderer != null) glSurfaceView.mRenderer.setScaleZ(scale);
            }
        });

        // Rotation
        createVerticalSlider(controls, "R", 360, 0, new SliderListener() {
            @Override
            public void onProgress(int progress) {
               if(glSurfaceView.mRenderer != null) glSurfaceView.mRenderer.setRotation(progress);
            }
        });

        // Translation
        createVerticalSlider(controls, "TX", 200, 100, new SliderListener() { // -1.0 to 1.0
            @Override
            public void onProgress(int progress) {
               float t = (progress - 100) / 100.0f;
               if(glSurfaceView.mRenderer != null) glSurfaceView.mRenderer.setTranslationX(t);
            }
        });
        createVerticalSlider(controls, "TY", 200, 100, new SliderListener() {
             @Override
             public void onProgress(int progress) {
                float t = (progress - 100) / 100.0f;
                if(glSurfaceView.mRenderer != null) glSurfaceView.mRenderer.setTranslationY(t);
             }
        });
        createVerticalSlider(controls, "TZ", 200, 100, new SliderListener() {
             @Override
             public void onProgress(int progress) {
                float t = (progress - 100) / 100.0f;
                if(glSurfaceView.mRenderer != null) glSurfaceView.mRenderer.setTranslationZ(t);
             }
        });

        // Reset Button
        android.widget.Button resetButton = new android.widget.Button(this);
        resetButton.setText("Reset");
        resetButton.setTextColor(android.graphics.Color.WHITE);
        resetButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(glSurfaceView.mRenderer != null) {
                    glSurfaceView.mRenderer.reset();
                }
            }
        });
        // Control layout
        controls.addView(resetButton);

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT
        );
        params.gravity = android.view.Gravity.BOTTOM | android.view.Gravity.RIGHT;
        frameLayout.addView(controls, params);

        setContentView(frameLayout);
    }

    @Override
    protected void onResume() {
        super.onResume();
        
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 0);
            return;
        }

        if (session == null) {
            try {
                session = new com.google.ar.core.Session(this);
                com.google.ar.core.Config config = new com.google.ar.core.Config(session);
                config.setInstantPlacementMode(com.google.ar.core.Config.InstantPlacementMode.LOCAL_Y_UP);
                session.configure(config);
                glSurfaceView.mRenderer.setSession(session);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        if (session != null) {
            try {
                session.resume();
            } catch (CameraNotAvailableException e) {
                throw new RuntimeException(e);
            }
        }
        glSurfaceView.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (session != null) {
            session.pause();
        }
        glSurfaceView.onPause();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
        }
    }
    interface SliderListener {
        void onProgress(int progress);
    }

    private int dpToPx(int dp) {
        return (int) (dp * getResources().getDisplayMetrics().density);
    }

    private void createVerticalSlider(android.widget.LinearLayout parent, String label, int max, int defaultVal, final SliderListener listener) {
        android.widget.LinearLayout container = new android.widget.LinearLayout(this);
        container.setOrientation(android.widget.LinearLayout.VERTICAL);
        container.setGravity(android.view.Gravity.CENTER_HORIZONTAL);
        
        // Label
        android.widget.TextView textView = new android.widget.TextView(this);
        textView.setText(label);
        textView.setTextColor(android.graphics.Color.WHITE);
        textView.setGravity(android.view.Gravity.CENTER);
        
        // Wrapper for rotation
        android.widget.FrameLayout wrapper = new android.widget.FrameLayout(this);
        int w = dpToPx(30);
        int h = dpToPx(200);
        wrapper.setLayoutParams(new android.widget.LinearLayout.LayoutParams(w, h));
        
        android.widget.SeekBar seekBar = new android.widget.SeekBar(this);
        seekBar.setMax(max);
        seekBar.setProgress(defaultVal);
        
        android.widget.FrameLayout.LayoutParams lp = new android.widget.FrameLayout.LayoutParams(h, w);
        lp.gravity = android.view.Gravity.CENTER;
        seekBar.setLayoutParams(lp);
        
        seekBar.setRotation(270);
        seekBar.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(android.widget.SeekBar seekBar, int progress, boolean fromUser) {
                listener.onProgress(progress);
            }
            @Override public void onStartTrackingTouch(android.widget.SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(android.widget.SeekBar seekBar) {}
        });
        
        wrapper.addView(seekBar);

        container.addView(wrapper);
        container.addView(textView);
        
        parent.addView(container);
    }
}