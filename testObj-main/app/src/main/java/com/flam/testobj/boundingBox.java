package com.flam.testobj;
import android.opengl.GLES30;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

public class boundingBox {

    private FloatBuffer vertexBuffer;
    private FloatBuffer colorBuffer;
    private ShortBuffer drawListBuffer;

    static final int COORDS_PER_VERTEX = 3;
    static final int COLORS_PER_VERTEX = 4;

    // 24 vertices (4 per face * 6 faces)
    static float cubeCoords[] = {
            // Front face
            -0.5f,  1.0f,  0.5f,
            -0.5f,  0.0f,  0.5f,
             0.5f,  0.0f,  0.5f,
             0.5f,  1.0f,  0.5f,
            // Back face
             0.5f,  1.0f, -0.5f,
             0.5f,  0.0f, -0.5f,
            -0.5f,  0.0f, -0.5f,
            -0.5f,  1.0f, -0.5f,
            // Left face
            -0.5f,  1.0f, -0.5f,
            -0.5f,  0.0f, -0.5f,
            -0.5f,  0.0f,  0.5f,
            -0.5f,  1.0f,  0.5f,
            // Right face
             0.5f,  1.0f,  0.5f,
             0.5f,  0.0f,  0.5f,
             0.5f,  0.0f, -0.5f,
             0.5f,  1.0f, -0.5f,
            // Top face
            -0.5f,  1.0f, -0.5f,
            -0.5f,  1.0f,  0.5f,
             0.5f,  1.0f,  0.5f,
             0.5f,  1.0f, -0.5f,
            // Bottom face
            -0.5f,  0.0f,  0.5f,
            -0.5f,  0.0f, -0.5f,
             0.5f,  0.0f, -0.5f,
             0.5f,  0.0f,  0.5f
    };

    // Colors for each vertex (face), translucent
    static float colors[] = {
            // Front face (Red)
            1.0f, 0.0f, 0.0f, 0.5f,
            1.0f, 0.0f, 0.0f, 0.5f,
            1.0f, 0.0f, 0.0f, 0.5f,
            1.0f, 0.0f, 0.0f, 0.5f,
            // Back face (Green)
            0.0f, 1.0f, 0.0f, 0.5f,
            0.0f, 1.0f, 0.0f, 0.5f,
            0.0f, 1.0f, 0.0f, 0.5f,
            0.0f, 1.0f, 0.0f, 0.5f,
            // Left face (Blue)
            0.0f, 0.0f, 1.0f, 0.5f,
            0.0f, 0.0f, 1.0f, 0.5f,
            0.0f, 0.0f, 1.0f, 0.5f,
            0.0f, 0.0f, 1.0f, 0.5f,
            // Right face (Yellow)
            1.0f, 1.0f, 0.0f, 0.5f,
            1.0f, 1.0f, 0.0f, 0.5f,
            1.0f, 1.0f, 0.0f, 0.5f,
            1.0f, 1.0f, 0.0f, 0.5f,
            // Top face (Cyan)
            0.0f, 1.0f, 1.0f, 0.5f,
            0.0f, 1.0f, 1.0f, 0.5f,
            0.0f, 1.0f, 1.0f, 0.5f,
            0.0f, 1.0f, 1.0f, 0.5f,
            // Bottom face (Magenta)
            1.0f, 0.0f, 1.0f, 0.5f,
            1.0f, 0.0f, 1.0f, 0.5f,
            1.0f, 0.0f, 1.0f, 0.5f,
            1.0f, 0.0f, 1.0f, 0.5f
    };

    private final short drawOrder[] = {
            0, 1, 2, 0, 2, 3,    // Front
            4, 5, 6, 4, 6, 7,    // Back
            8, 9, 10, 8, 10, 11, // Left
            12, 13, 14, 12, 14, 15, // Right
            16, 17, 18, 16, 18, 19, // Top
            20, 21, 22, 20, 22, 23  // Bottom
    };

    private final int vertexStride = COORDS_PER_VERTEX * 4;
    private final int colorStride = COLORS_PER_VERTEX * 4;

    public boundingBox() {
        ByteBuffer bb = ByteBuffer.allocateDirect(cubeCoords.length * 4);
        bb.order(ByteOrder.nativeOrder());
        vertexBuffer = bb.asFloatBuffer();
        vertexBuffer.put(cubeCoords);
        vertexBuffer.position(0);

        ByteBuffer cb = ByteBuffer.allocateDirect(colors.length * 4);
        cb.order(ByteOrder.nativeOrder());
        colorBuffer = cb.asFloatBuffer();
        colorBuffer.put(colors);
        colorBuffer.position(0);

        ByteBuffer dlb = ByteBuffer.allocateDirect(drawOrder.length * 2);
        dlb.order(ByteOrder.nativeOrder());
        drawListBuffer = dlb.asShortBuffer();
        drawListBuffer.put(drawOrder);
        drawListBuffer.position(0);
    }

    public void draw(int positionHandle, int colorHandle, int mvpMatrixHandle, float[] mvpMatrix) {
        GLES30.glVertexAttribPointer(
                positionHandle, COORDS_PER_VERTEX,
                GLES30.GL_FLOAT, false,
                vertexStride, vertexBuffer);
        GLES30.glEnableVertexAttribArray(positionHandle);

        GLES30.glVertexAttribPointer(
                colorHandle, COLORS_PER_VERTEX,
                GLES30.GL_FLOAT, false,
                colorStride, colorBuffer);
        GLES30.glEnableVertexAttribArray(colorHandle);

        GLES30.glUniformMatrix4fv(mvpMatrixHandle, 1, false, mvpMatrix, 0);

        GLES30.glDrawElements(
                GLES30.GL_TRIANGLES, drawOrder.length,
                GLES30.GL_UNSIGNED_SHORT, drawListBuffer);

        GLES30.glDisableVertexAttribArray(positionHandle);
        GLES30.glDisableVertexAttribArray(colorHandle);
    }
}
