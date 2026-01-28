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

class CaptureManager(private val context: Context) {

    private val gson = GsonBuilder().setPrettyPrinting().create()
    private var sequenceId = 0
    private var currentDir: File? = null
    private val annotations = mutableListOf<AnnotationEntry>()
    private val ioDispatcher = kotlinx.coroutines.Dispatchers.IO
    private val scope = kotlinx.coroutines.MainScope()

    fun startNewSequence(category: String) {
        val storageDir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        val categoryFolder = File(storageDir, category)
        categoryFolder.mkdirs()
        
        sequenceId = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date()).hashCode().coerceAtLeast(0)
        currentDir = File(categoryFolder, "seq_$sequenceId")
        currentDir?.mkdirs()
        annotations.clear()
    }

    fun saveFrame(bitmap: Bitmap, entry: AnnotationEntry) {
        if (currentDir == null) startNewSequence()

        val imageFile = File(currentDir, entry.image)
        
        // Save image on background thread
        scope.launch(ioDispatcher) {
            try {
                FileOutputStream(imageFile).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                // Reuse bitmap if possible or let GC handle it
            }
        }
        
        synchronized(annotations) {
            annotations.add(entry)
        }
    }

    fun finishSequence() {
        saveAnnotationsLocal()
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
