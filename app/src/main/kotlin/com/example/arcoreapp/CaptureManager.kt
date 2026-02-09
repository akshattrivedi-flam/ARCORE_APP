package com.example.arcoreapp

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import com.google.gson.GsonBuilder
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*

import com.google.ar.core.RecordingConfig
import com.google.ar.core.Session
import android.net.Uri

class CaptureManager(private val context: Context) {

    private val gson = GsonBuilder().setPrettyPrinting().create()
    private var currentDir: File? = null
    private var rawFramesDir: File? = null
    private var annotatedFramesDir: File? = null
    private val annotations = mutableListOf<AnnotationEntry>()
    private val ioDispatcher = kotlinx.coroutines.Dispatchers.IO
    private val scope = kotlinx.coroutines.MainScope()
    
    private val prefs = context.getSharedPreferences("objectron_prefs", Context.MODE_PRIVATE)

    fun startNewSequence(category: String, session: Session? = null) {
        val storageDir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        val categoryFolder = File(storageDir, category)
        categoryFolder.mkdirs()
        
        // Get index (next number)
        val count = getCount(category) + 1
        val indexStr = String.format("%02d", count)
        
        // Format: video_01_red
        currentDir = File(categoryFolder, "video_${indexStr}_$category")
        currentDir?.mkdirs()
        
        // Unified Resolution Folders
        rawFramesDir = File(currentDir, "raw_frames")
        rawFramesDir?.mkdirs()
        annotatedFramesDir = File(currentDir, "annotated_frames")
        annotatedFramesDir?.mkdirs()
        
        annotations.clear()

        // Start ARCore Session Recording (Raw Sensor Feed)
        // Note: This remains in sensor resolution (640x480) as it's a raw dump.
        session?.let {
            val videoFile = File(currentDir, "video_raw.mp4")
            try {
                val recordingConfig = RecordingConfig(it)
                    .setMp4DatasetUri(Uri.fromFile(videoFile))
                    .setAutoStopOnPause(true)
                it.startRecording(recordingConfig)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun saveFrame(bitmap: Bitmap, entry: AnnotationEntry) {
        val rawDir = this.rawFramesDir ?: return
        val annDir = this.annotatedFramesDir ?: return

        val rawFile = File(rawDir, entry.image)
        val annFile = File(annDir, entry.image)
        
        // 1. Create Annotated Version
        val annotatedBitmap = drawBoxOnBitmap(bitmap, entry.keypoints_2d)
        
        scope.launch(ioDispatcher) {
            try {
                // Save Clean Frame
                FileOutputStream(rawFile).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }
                // Save Annotated Frame
                FileOutputStream(annFile).use { out ->
                    annotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }
                // Clean up annotated bitmap memory
                annotatedBitmap.recycle()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        
        synchronized(annotations) {
            annotations.add(entry)
        }
    }

    /**
     * Helper to draw the Objectron 3D Bounding Box onto a Bitmap.
     */
    private fun drawBoxOnBitmap(bitmap: Bitmap, keypoints: List<List<Float>>): Bitmap {
        val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = android.graphics.Canvas(result)
        val paint = android.graphics.Paint().apply {
            color = android.graphics.Color.GREEN
            strokeWidth = 3f
            style = android.graphics.Paint.Style.STROKE
        }
        
        val w = result.width.toFloat()
        val h = result.height.toFloat()
        
        // Safety check
        if (keypoints.size < 9) return result
        
        fun getP(i: Int) = android.graphics.PointF(keypoints[i][0] * w, keypoints[i][1] * h)
        
        // 0: Center (Dot)
        val c = getP(0)
        canvas.drawCircle(c.x, c.y, 4f, paint)

        // Faces mapping (Objectron Standard)
        // 1-4: Front, 5-8: Back
        val p = Array(9) { getP(it) }
        
        // Front Face
        canvas.drawLine(p[1].x, p[1].y, p[2].x, p[2].y, paint)
        canvas.drawLine(p[2].x, p[2].y, p[3].x, p[3].y, paint)
        canvas.drawLine(p[3].x, p[3].y, p[4].x, p[4].y, paint)
        canvas.drawLine(p[4].x, p[4].y, p[1].x, p[1].y, paint)
        
        // Back Face
        canvas.drawLine(p[5].x, p[5].y, p[6].x, p[6].y, paint)
        canvas.drawLine(p[6].x, p[6].y, p[7].x, p[7].y, paint)
        canvas.drawLine(p[7].x, p[7].y, p[8].x, p[8].y, paint)
        canvas.drawLine(p[8].x, p[8].y, p[5].x, p[5].y, paint)
        
        // Connecting Lines
        for (i in 1..4) {
            canvas.drawLine(p[i].x, p[i].y, p[i+4].x, p[i+4].y, paint)
        }
        
        return result
    }

    fun finishSequence(category: String, session: Session? = null) {
        session?.let {
            try {
                it.stopRecording()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        saveAnnotationsLocal()
        incrementCount(category)
    }

    private fun incrementCount(category: String) {
        val current = getCount(category)
        prefs.edit().putInt("count_$category", current + 1).apply()
    }

    fun getCount(category: String): Int {
        return prefs.getInt("count_$category", 0)
    }

    private fun saveAnnotationsLocal() {
        scope.launch(ioDispatcher) {
            val jsonFile = File(currentDir, "annotations.json")
            synchronized(annotations) {
                jsonFile.writeText(gson.toJson(annotations))
            }
        }
    }

    fun getCapturePath(): String {
        return currentDir?.absolutePath ?: "N/A"
    }
}
