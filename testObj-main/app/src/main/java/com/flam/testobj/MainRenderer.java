package com.flam.testobj;

import android.content.Context;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MainRenderer implements GLSurfaceView.Renderer {

    Context context;

    private boundingBox mBox;
    private int mProgram;
    private BackgroundRenderer backgroundRenderer = new BackgroundRenderer();
    private com.google.ar.core.Session session;
    private com.google.ar.core.Anchor anchor;
    private final java.util.concurrent.ArrayBlockingQueue<android.view.MotionEvent> queuedSingleTaps = new java.util.concurrent.ArrayBlockingQueue<>(16);

    private final float[] mMVPMatrix = new float[16];
    private final float[] mProjectionMatrix = new float[16];
    private final float[] mViewMatrix = new float[16];
    private final float[] mAnchorMatrix = new float[16];

    private int positionHandle;
    private int colorHandle;
    private int mvpMatrixHandle;

    private final String vertexShaderCode =
            "uniform mat4 uMVPMatrix;" +
            "attribute vec4 vPosition;" +
            "attribute vec4 aColor;" +
            "varying vec4 vColor;" +
            "void main() {" +
            "  gl_Position = uMVPMatrix * vPosition;" +
            "  vColor = aColor;" +
            "}";

    private final String fragmentShaderCode =
            "precision mediump float;" +
            "varying vec4 vColor;" +
            "void main() {" +
            "  gl_FragColor = vColor;" +
            "}";

    public MainRenderer(Context context) {
        this.context = context;
    }

    private volatile float mScaleX = 0.1f;
    private volatile float mScaleY = 0.1f;
    private volatile float mScaleZ = 0.1f;
    private volatile float mRotationY = 0.0f;
    private volatile float mTranslationX = 0.0f;
    private volatile float mTranslationY = 0.0f;
    private volatile float mTranslationZ = 0.0f;

    public void setSession(com.google.ar.core.Session session) {
        this.session = session;
    }

    public void setScaleX(float scale) {
        mScaleX = scale;
    }
    public void setScaleY(float scale) {
        mScaleY = scale;
    }
    public void setScaleZ(float scale) {
        mScaleZ = scale;
    }

    public void setRotation(float angle) {
        mRotationY = angle;
    }

    public void setTranslationX(float t) {
        mTranslationX = t;
    }
    public void setTranslationY(float t) {
        mTranslationY = t;
    }
    public void setTranslationZ(float t) {
        mTranslationZ = t;
    }

    public void onTouch(android.view.MotionEvent event) {
        queuedSingleTaps.offer(event);
    }

    public static int loadShader(int type, String shaderCode){
        int shader = GLES30.glCreateShader(type);
        GLES30.glShaderSource(shader, shaderCode);
        GLES30.glCompileShader(shader);
        return shader;
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES30.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        GLES30.glEnable(GLES30.GL_BLEND);
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA);

        mBox = new boundingBox();
        backgroundRenderer.createOnGlThread();

        int vertexShader = loadShader(GLES30.GL_VERTEX_SHADER, vertexShaderCode);
        int fragmentShader = loadShader(GLES30.GL_FRAGMENT_SHADER, fragmentShaderCode);

        mProgram = GLES30.glCreateProgram();
        GLES30.glAttachShader(mProgram, vertexShader);
        GLES30.glAttachShader(mProgram, fragmentShader);
        GLES30.glLinkProgram(mProgram);
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES30.glViewport(0, 0, width, height);
        int rotation = android.view.Surface.ROTATION_0;
        if (context != null) {
            android.view.WindowManager wm = (android.view.WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
            if (wm != null) {
                rotation = wm.getDefaultDisplay().getRotation();
            }
        }
        if (session != null) {
            session.setDisplayGeometry(rotation, width, height);
        }
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);

        if (session == null) {
            return;
        }

        try {
            session.setCameraTextureName(backgroundRenderer.getTextureId());
            com.google.ar.core.Frame frame = session.update();
            com.google.ar.core.Camera camera = frame.getCamera();

            backgroundRenderer.draw(frame);

            handleTap(frame, camera);

            if (camera.getTrackingState() == com.google.ar.core.TrackingState.TRACKING && anchor != null) {
                camera.getProjectionMatrix(mProjectionMatrix, 0, 0.1f, 100.0f);
                camera.getViewMatrix(mViewMatrix, 0);
                
                anchor.getPose().toMatrix(mAnchorMatrix, 0);
                Matrix.translateM(mAnchorMatrix, 0, mTranslationX, mTranslationY, mTranslationZ);
                Matrix.rotateM(mAnchorMatrix, 0, mRotationY, 0f, 1f, 0f);
                Matrix.scaleM(mAnchorMatrix, 0, mScaleX, mScaleY, mScaleZ);

                Matrix.multiplyMM(mMVPMatrix, 0, mViewMatrix, 0, mAnchorMatrix, 0);
                Matrix.multiplyMM(mMVPMatrix, 0, mProjectionMatrix, 0, mMVPMatrix, 0);

                drawBox();
            }

        } catch (Throwable t) {
            // Avoid crashing on session error
        }
    }

    public void reset() {
        if (anchor != null) {
            anchor.detach();
            anchor = null;
        }
    }

    private void handleTap(com.google.ar.core.Frame frame, com.google.ar.core.Camera camera) {
        android.view.MotionEvent tap = queuedSingleTaps.poll();
        if (tap != null && camera.getTrackingState() == com.google.ar.core.TrackingState.TRACKING) {
            if (anchor != null) {
                return;
            }
            for (com.google.ar.core.HitResult hit : frame.hitTest(tap)) {

                com.google.ar.core.Trackable trackable = hit.getTrackable();
                if ((trackable instanceof com.google.ar.core.Plane
                        && ((com.google.ar.core.Plane) trackable).isPoseInPolygon(hit.getHitPose())
                        && (com.google.ar.core.Plane.Type.HORIZONTAL_UPWARD_FACING == ((com.google.ar.core.Plane) trackable).getType()))
                        || (trackable instanceof com.google.ar.core.Point
                        && ((com.google.ar.core.Point) trackable).getOrientationMode()
                        == com.google.ar.core.Point.OrientationMode.ESTIMATED_SURFACE_NORMAL)
                         || (trackable instanceof com.google.ar.core.InstantPlacementPoint)) {
                    
                    anchor = hit.createAnchor();
                    break;
                }
            }
        }
    }

    private void drawBox() {
        GLES30.glUseProgram(mProgram);
        positionHandle = GLES30.glGetAttribLocation(mProgram, "vPosition");
        colorHandle = GLES30.glGetAttribLocation(mProgram, "aColor");
        mvpMatrixHandle = GLES30.glGetUniformLocation(mProgram, "uMVPMatrix");

        mBox.draw(positionHandle, colorHandle, mvpMatrixHandle, mMVPMatrix);
    }

    public void onDestroy() {
        if (session != null) {
            session.close();
            session = null;
        }
    }
}
