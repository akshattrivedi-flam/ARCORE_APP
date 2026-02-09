package com.example.arcoreapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object YuvToRgbConverter {

    fun yuvToBitmap(image: Image): Bitmap? {
        if (image.format != ImageFormat.YUV_420_888) {
            throw IllegalArgumentException("Image must be in YUV_420_888 format")
        }

        val nv21 = yuv420888ToNv21(image)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun yuv420888ToNv21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 2
        val nv21 = ByteArray(ySize + uvSize)

        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val rowStride = image.planes[0].rowStride
        val pixelStride = image.planes[1].pixelStride

        var pos = 0

        if (rowStride == width) { // Likely compatible
            yBuffer.get(nv21, 0, ySize)
            pos = ySize
        } else {
            // Row-by-row copy for padding handling
            var yPos = 0
            for (row in 0 until height) {
                yBuffer.position(yPos)
                yBuffer.get(nv21, pos, width)
                pos += width
                yPos += rowStride
            }
        }

        val uvHeight = height / 2
        val uvWidth = width / 2
        
        // Handling UV planes for NV21 (V first, then U interleaved VUVU...)
        // But Input might be separate planar or semi-planar.
        // We interleave manually to be safe.
        
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val vIndex = row * image.planes[2].rowStride + col * pixelStride
                val uIndex = row * image.planes[1].rowStride + col * pixelStride
                
                nv21[pos++] = vBuffer.get(vIndex)
                nv21[pos++] = uBuffer.get(uIndex)
            }
        }

        return nv21
    }
}
