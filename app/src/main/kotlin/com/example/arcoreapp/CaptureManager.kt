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
    private var sequenceId = 0
    private var currentDir: File? = null
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
        annotations.clear()

        // Start ARCore Session Recording (Raw Feed)
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
        val currentDir = this.currentDir ?: return

        val imageFile = File(currentDir, entry.image)
        
        scope.launch(ioDispatcher) {
            try {
                FileOutputStream(imageFile).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        
        synchronized(annotations) {
            annotations.add(entry)
        }
    }

    fun finishSequence(category: String, session: Session? = null) {
        // Stop ARCore Recording
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
