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

    /**
     * Converts an android.media.Image in YUV_420_888 format into a standard Android Bitmap (ARGB_8888).
     * 
     * This process involves:
     * 1. Extracting the raw Y, U, and V plane buffers.
     * 2. Converting these planes into a single NV21 byte array (standard YUV format).
     * 3. Using Android's YuvImage class to compress the NV21 data into a JPEG.
     * 4. Decoding the JPEG bytes back into a Bitmap.
     * 
     * @param image The input image from ARCore or Camera2 API.
     * @return A Bitmap representation of the image, or null if conversion fails.
     */
    fun yuvToBitmap(image: Image): Bitmap? {
        if (image.format != ImageFormat.YUV_420_888) {
            throw IllegalArgumentException("Image must be in YUV_420_888 format")
        }

        // 1. Convert complex multi-plane YUV_420 translation to linear NV21 byte array
        val nv21 = yuv420888ToNv21(image)

        // 2. Wrap NV21 data in YuvImage helper
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        
        // 3. Compress to JPEG (High quality 100)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        
        // 4. Decode JPEG to Bitmap (Result is ARGB_8888 by default)
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Converts complex planar YUV_420_888 data into the semi-planar NV21 format.
     * 
     * YUV_420_888 is a generic format that can be Planar (Y, U, V separate) or 
     * Semi-Planar (Y, UV interleaved). It also supports "strides" (padding bytes at the end of rows).
     * 
     * NV21 is a specific standard:
     * - A full block of Y (grayscale) data first.
     * - Followed by interleaved V and U bytes (V, U, V, U...) for 2x2 subsampled color.
     * 
     * This function manually copies bytes pixel-by-pixel to ensure stride compatibility on all devices.
     */
    private fun yuv420888ToNv21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        
        // NV21 Size = Y_Size (w*h) + UV_Size (w*h/2) = 1.5 * w * h
        val ySize = width * height
        val uvSize = width * height / 2
        val nv21 = ByteArray(ySize + uvSize)

        val yBuffer = image.planes[0].buffer // Y Plane (Luma)
        val uBuffer = image.planes[1].buffer // U Plane (Chroma Blue)
        val vBuffer = image.planes[2].buffer // V Plane (Chroma Red)

        // Stride: The number of bytes in memory per row. Can be > width due to padding.
        val rowStride = image.planes[0].rowStride
        // PixelStride: Distance between adjacent pixel samples. 1 for planar, 2 for semi-planar UV.
        val pixelStride = image.planes[1].pixelStride

        var pos = 0

        // --- 1. Copy Y Plane (Luminance) ---
        if (rowStride == width) { 
            // Fast Path: Check if there is NO padding. We can bulk copy.
            yBuffer.get(nv21, 0, ySize)
            pos = ySize
        } else {
            // Safe Path: Device uses padding. Copy row-by-row, skipping padding bytes.
            var yPos = 0
            for (row in 0 until height) {
                yBuffer.position(yPos)
                yBuffer.get(nv21, pos, width) // Read 'width' bytes
                pos += width
                yPos += rowStride // Jump to next row start
            }
        }

        // --- 2. Copy UV Planes (Chrominance) ---
        // YUV 4:2:0 subsampling means 1 UV pair for every 2x2 block of pixels.
        // So UV dimensions are half of Width and Height.
        val uvHeight = height / 2
        val uvWidth = width / 2
        
        // We need to construct NV21 which format is: V, U, V, U...
        // We read from the source U and V buffers using their specific strides.
        
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                // Calculate correct index in the source buffers
                val vIndex = row * image.planes[2].rowStride + col * pixelStride
                val uIndex = row * image.planes[1].rowStride + col * pixelStride
                
                // NV21 expects V first, then U
                nv21[pos++] = vBuffer.get(vIndex)
                nv21[pos++] = uBuffer.get(uIndex)
            }
        }

        return nv21
    }
}
