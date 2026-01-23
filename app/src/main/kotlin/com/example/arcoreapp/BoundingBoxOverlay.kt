package com.example.arcoreapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private var points2d: List<List<Float>>? = null

    fun updatePoints(points: List<List<Float>>) {
        this.points2d = points
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val pts = points2d ?: return
        if (pts.size < 9) return

        // pts[0] is center, pts[1..8] are corners
        // Objectron order: 
        // 1-2-3-4 is front face
        // 5-6-7-8 is back face
        
        val w = width.toFloat()
        val h = height.toFloat()

        fun getX(i: Int) = pts[i][0] * w
        fun getY(i: Int) = pts[i][1] * h

        // Draw Front face
        drawLine(canvas, pts, 1, 2, w, h)
        drawLine(canvas, pts, 2, 3, w, h)
        drawLine(canvas, pts, 3, 4, w, h)
        drawLine(canvas, pts, 4, 1, w, h)

        // Draw Back face
        drawLine(canvas, pts, 5, 6, w, h)
        drawLine(canvas, pts, 6, 7, w, h)
        drawLine(canvas, pts, 7, 8, w, h)
        drawLine(canvas, pts, 8, 5, w, h)

        // Draw connecting lines
        drawLine(canvas, pts, 1, 5, w, h)
        drawLine(canvas, pts, 2, 6, w, h)
        drawLine(canvas, pts, 3, 7, w, h)
        drawLine(canvas, pts, 4, 8, w, h)
        
        // Draw center point
        canvas.drawCircle(getX(0), getY(0), 10f, paint)
    }

    private fun drawLine(canvas: Canvas, pts: List<List<Float>>, i: Int, j: Int, w: Float, h: Float) {
        canvas.drawLine(pts[i][0] * w, pts[i][1] * h, pts[j][0] * w, pts[j][1] * h, paint)
    }
}
