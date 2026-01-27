package com.flam.testobj;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;

public class MainSurfaceView extends GLSurfaceView {
    MainRenderer mRenderer;

    public MainSurfaceView(Context context) {
        super(context);
        setEGLContextClientVersion(3);
        mRenderer = new MainRenderer(context);
        setRenderer(mRenderer);
        setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    }


    public void onDestroy(){
        mRenderer.onDestroy();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (mRenderer != null) {
            mRenderer.onTouch(event);
        }
        return super.onTouchEvent(event);
    }
}