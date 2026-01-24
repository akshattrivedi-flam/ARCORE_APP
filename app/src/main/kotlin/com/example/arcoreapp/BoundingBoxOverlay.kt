package com.example.arcoreapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

import android.graphics.Path

class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val strokePaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val fillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val borderPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        isAntiAlias = true
        alpha = 180
    }

    private val facePath = Path()

    private var points2d: List<List<Float>>? = null

    fun updatePoints(points: List<List<Float>>) {
        this.points2d = points
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val pts = points2d ?: return
        if (pts.size < 9) return

        val w = width.toFloat()
        val h = height.toFloat()

        // Face definitions (1-based index)
        // 1-2-3-4: Front
        // 5-6-7-8: Back
        // 1-2-6-5: Bottom
        // 4-3-7-8: Top
        // 1-4-8-5: Left
        // 2-3-7-6: Right

        // Draw faces with specific colors (40/255 opacity for translucency)
        val opacity = 80
        drawFace(canvas, pts, listOf(1, 2, 3, 4), Color.RED, opacity, w, h)    // Front
        drawFace(canvas, pts, listOf(5, 6, 7, 8), Color.RED, opacity, w, h)    // Back
        
        drawFace(canvas, pts, listOf(1, 4, 8, 5), Color.GREEN, opacity, w, h)  // Left
        drawFace(canvas, pts, listOf(2, 3, 7, 6), Color.GREEN, opacity, w, h)  // Right
        
        drawFace(canvas, pts, listOf(4, 3, 7, 8), Color.BLUE, opacity, w, h)   // Top
        drawFace(canvas, pts, listOf(1, 2, 6, 5), Color.YELLOW, opacity, w, h) // Bottom

        // Draw Wireframe
        strokePaint.color = Color.WHITE
        strokePaint.strokeWidth = 4f
        // Front
        drawLine(canvas, pts, 1, 2, w, h)
        drawLine(canvas, pts, 2, 3, w, h)
        drawLine(canvas, pts, 3, 4, w, h)
        drawLine(canvas, pts, 4, 1, w, h)
        // Back
        drawLine(canvas, pts, 5, 6, w, h)
        drawLine(canvas, pts, 6, 7, w, h)
        drawLine(canvas, pts, 7, 8, w, h)
        drawLine(canvas, pts, 8, 5, w, h)
        // Connections
        drawLine(canvas, pts, 1, 5, w, h)
        drawLine(canvas, pts, 2, 6, w, h)
        drawLine(canvas, pts, 3, 7, w, h)
        drawLine(canvas, pts, 4, 8, w, h)

        // Center point
        strokePaint.style = Paint.Style.FILL
        canvas.drawCircle(pts[0][0] * w, pts[0][1] * h, 10f, strokePaint)
        strokePaint.style = Paint.Style.STROKE
    }

    private fun drawFace(canvas: Canvas, pts: List<List<Float>>, indices: List<Int>, color: Int, opacity: Int, w: Float, h: Float) {
        fillPaint.color = color
        fillPaint.alpha = opacity
        borderPaint.color = color
        
        facePath.reset()
        facePath.moveTo(pts[indices[0]][0] * w, pts[indices[0]][1] * h)
        for (i in 1 until indices.size) {
            facePath.lineTo(pts[indices[i]][0] * w, pts[indices[i]][1] * h)
        }
        facePath.close()
        
        canvas.drawPath(facePath, fillPaint)
        canvas.drawPath(facePath, borderPaint)
    }

    private fun drawLine(canvas: Canvas, pts: List<List<Float>>, i: Int, j: Int, w: Float, h: Float) {
        canvas.drawLine(pts[i][0] * w, pts[i][1] * h, pts[j][0] * w, pts[j][1] * h, strokePaint)
    }
}
