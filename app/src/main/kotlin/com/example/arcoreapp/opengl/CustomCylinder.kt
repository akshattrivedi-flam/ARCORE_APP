package com.example.arcoreapp.opengl

import android.opengl.GLES30
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.sin

class CustomCylinder(segments: Int = 30) {

    private val vertexBuffer: FloatBuffer
    private val vertexCount: Int

    init {
        val vertices = mutableListOf<Float>()
        val radius = 0.5f
        val height = 1.0f

        // Generate cylinder vertices (sides only for wireframe)
        for (i in 0..segments) {
            val angle = 2.0f * Math.PI.toFloat() * i / segments
            val x = radius * cos(angle.toDouble()).toFloat()
            val z = radius * sin(angle.toDouble()).toFloat()

            // Bottom point
            vertices.add(x)
            vertices.add(0.0f)
            vertices.add(z)

            // Top point
            vertices.add(x)
            vertices.add(height)
            vertices.add(z)
        }

        vertexCount = vertices.size / 3
        val bb = ByteBuffer.allocateDirect(vertices.size * 4)
        bb.order(ByteOrder.nativeOrder())
        vertexBuffer = bb.asFloatBuffer()
        vertices.forEach { vertexBuffer.put(it) }
        vertexBuffer.position(0)
    }

    fun draw(positionHandle: Int, mvpMatrixHandle: Int, mvpMatrix: FloatArray) {
        GLES30.glVertexAttribPointer(
            positionHandle, 3,
            GLES30.GL_FLOAT, false,
            0, vertexBuffer
        )
        GLES30.glEnableVertexAttribArray(positionHandle)

        GLES30.glUniformMatrix4fv(mvpMatrixHandle, 1, false, mvpMatrix, 0)

        // Draw as a triangle strip to create the cylinder shell
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, vertexCount)

        GLES30.glDisableVertexAttribArray(positionHandle)
    }
}
