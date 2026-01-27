package com.flam.testobj;

import android.opengl.GLES11Ext;
import android.opengl.GLES30;

import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class BackgroundRenderer {
    private static final String VERTEX_SHADER =
            "attribute vec4 a_Position;\n" +
            "attribute vec2 a_TexCoord;\n" +
            "varying vec2 v_TexCoord;\n" +
            "void main() {\n" +
            "   gl_Position = a_Position;\n" +
            "   v_TexCoord = a_TexCoord;\n" +
            "}";

    private static final String FRAGMENT_SHADER =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision mediump float;\n" +
            "varying vec2 v_TexCoord;\n" +
            "uniform samplerExternalOES s_Texture;\n" +
            "void main() {\n" +
            "    gl_FragColor = texture2D(s_Texture, v_TexCoord);\n" +
            "}";

    private static final float[] QUAD_COORDS = new float[]{
            -1.0f, -1.0f,
            -1.0f, +1.0f,
            +1.0f, -1.0f,
            +1.0f, +1.0f,
    };

    private static final float[] QUAD_TEXCOORDS = new float[]{
            0.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 1.0f,
            1.0f, 0.0f,
    };

    private int program;
    private int textureId;
    private int positionAttribute;
    private int texCoordAttribute;
    private int textureUniform;

    private FloatBuffer quadCoords;
    private FloatBuffer quadTexCoords;

    public int getTextureId() {
        return textureId;
    }

    public void createOnGlThread() {
        // Generate the background texture.
        int[] textures = new int[1];
        GLES30.glGenTextures(1, textures, 0);
        textureId = textures[0];
        int textureTarget = GLES11Ext.GL_TEXTURE_EXTERNAL_OES;
        GLES30.glBindTexture(textureTarget, textureId);
        GLES30.glTexParameteri(textureTarget, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(textureTarget, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(textureTarget, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(textureTarget, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);

        int vertexShader = MainRenderer.loadShader(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER);
        int fragmentShader = MainRenderer.loadShader(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);

        program = GLES30.glCreateProgram();
        GLES30.glAttachShader(program, vertexShader);
        GLES30.glAttachShader(program, fragmentShader);
        GLES30.glLinkProgram(program);

        positionAttribute = GLES30.glGetAttribLocation(program, "a_Position");
        texCoordAttribute = GLES30.glGetAttribLocation(program, "a_TexCoord");
        textureUniform = GLES30.glGetUniformLocation(program, "s_Texture");

        quadCoords = ByteBuffer.allocateDirect(QUAD_COORDS.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        quadCoords.put(QUAD_COORDS);
        quadCoords.position(0);

        quadTexCoords = ByteBuffer.allocateDirect(QUAD_TEXCOORDS.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        quadTexCoords.put(QUAD_TEXCOORDS);
        quadTexCoords.position(0);
    }

    public void draw(Frame frame) {
        if (frame.hasDisplayGeometryChanged()) {
            quadCoords.position(0);
            quadTexCoords.position(0);
            frame.transformCoordinates2d(
                    Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                    quadCoords,
                    Coordinates2d.TEXTURE_NORMALIZED,
                    quadTexCoords);
        }

        GLES30.glDisable(GLES30.GL_DEPTH_TEST);
        GLES30.glDepthMask(false);

        GLES30.glUseProgram(program);

        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES30.glUniform1i(textureUniform, 0);

        quadCoords.position(0);
        quadTexCoords.position(0);

        GLES30.glVertexAttribPointer(positionAttribute, 2, GLES30.GL_FLOAT, false, 0, quadCoords);
        GLES30.glVertexAttribPointer(texCoordAttribute, 2, GLES30.GL_FLOAT, false, 0, quadTexCoords);

        GLES30.glEnableVertexAttribArray(positionAttribute);
        GLES30.glEnableVertexAttribArray(texCoordAttribute);

        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4);

        GLES30.glDisableVertexAttribArray(positionAttribute);
        GLES30.glDisableVertexAttribArray(texCoordAttribute);

        GLES30.glDepthMask(true);
        GLES30.glEnable(GLES30.GL_DEPTH_TEST);
    }
}
