package com.example.arcoreapp

import android.opengl.Matrix

object MathUtils {

    /**
     * Projects a 3D point in world space to 2D image coordinates.
     * @param pointWorld 3D point in world space [x, y, z]
     * @param viewMatrix 4x4 View Matrix (World to Camera)
     * @param intrinsics Camera intrinsics (fx, fy, cx, cy, width, height)
     * @return floatArrayOf(x_norm, y_norm, depth)
     */
    fun projectPoint(
        pointWorld: FloatArray,
        viewMatrix: FloatArray,
        fx: Float, fy: Float, cx: Float, cy: Float,
        width: Int, height: Int
    ): FloatArray {
        // 1. World to Camera: P_camera = V * P_world
        val pointWorld4 = floatArrayOf(pointWorld[0], pointWorld[1], pointWorld[2], 1.0f)
        val pointCamera = FloatArray(4)
        Matrix.multiplyMV(pointCamera, 0, viewMatrix, 0, pointWorld4, 0)

        val x = pointCamera[0]
        val y = pointCamera[1]
        val z = pointCamera[2]

        // 2. Camera to Image Plane (using intrinsics)
        // X+ is Right, Y+ is Up, Z- is Forward (opposite of camera looking dir).
        // Depth is positive distance from camera
        val depth = -z 
        
        // Project to image space (0,0 is center of intrinsics)
        val u = fx * (x / depth) + cx
        
        // CRITICAL FIX: In camera space Y is UP, but in screen/image space Y is DOWN.
        // We must negate the Y term to correctly project from camera to image pixels.
        val v = cy - fy * (y / depth) 

        // Normalize to [0..1]
        val xNorm = u / width
        val yNorm = v / height

        return floatArrayOf(xNorm, yNorm, depth)
    }

    /**
     * Multiplies a 4x4 matrix by a 4x1 vector.
     */
    fun multiplyMV(matrix: FloatArray, vector: FloatArray): FloatArray {
        val result = FloatArray(4)
        Matrix.multiplyMV(result, 0, matrix, 0, vector, 0)
        return result
    }

    /**
     * Creates a model matrix from translation, rotation (euler), and scale.
     */
    fun createModelMatrix(
        tX: Float, tY: Float, tZ: Float,
        rYaw: Float, rPitch: Float, rRoll: Float,
        sX: Float, sY: Float, sZ: Float
    ): FloatArray {
        val modelMatrix = FloatArray(16)
        Matrix.setIdentityM(modelMatrix, 0)
        Matrix.translateM(modelMatrix, 0, tX, tY, tZ)
        Matrix.rotateM(modelMatrix, 0, rYaw, 0f, 1f, 0f)
        Matrix.rotateM(modelMatrix, 0, rPitch, 1f, 0f, 0f)
        Matrix.rotateM(modelMatrix, 0, rRoll, 0f, 0f, 1f)
        Matrix.scaleM(modelMatrix, 0, sX, sY, sZ)
        return modelMatrix
    }
}
