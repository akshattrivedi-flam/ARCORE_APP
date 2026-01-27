package com.example.arcoreapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Bundle
import android.view.PixelCopy
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.arcoreapp.databinding.ActivityMainBinding
import com.example.arcoreapp.opengl.ARRenderer
import com.google.ar.core.*
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var renderer: ARRenderer
    private var arSession: Session? = null
    
    private lateinit var captureManager: CaptureManager
    private var frameCount = 0

    // Transform properties (Manual fitting) - values in centimeters
    private var scaleX = 6.5f
    private var scaleY = 12f
    private var scaleZ = 6.5f
    private var rotY = 0f
    private var transX = 0f
    private var transY = 6f
    private var transZ = 0f

    private var isRecording = false
    private var isProcessingFrame = false
    private var lastFrameTime = 0L

    companion object {
        private const val CAMERA_PERMISSION_CODE = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        glSurfaceView = binding.glSurfaceView
        captureManager = CaptureManager(this)

        if (checkCameraPermission()) {
            setupAR()
        } else {
            requestCameraPermission()
        }
        setupControls()
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                setupAR()
            } else {
                Toast.makeText(this, "Camera permission is required for AR", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun setupAR() {
        renderer = ARRenderer(this)
        glSurfaceView.preserveEGLContextOnPause = true
        glSurfaceView.setEGLContextClientVersion(3)
        glSurfaceView.setRenderer(renderer)
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        try {
            arSession = Session(this)
            val config = Config(arSession)
            
            // Enable Depth for better anchoring
            if (arSession!!.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                config.depthMode = Config.DepthMode.AUTOMATIC
            }
            
            config.instantPlacementMode = Config.InstantPlacementMode.LOCAL_Y_UP
            config.focusMode = Config.FocusMode.AUTO
            config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            arSession!!.configure(config)
            
            renderer.setArSession(arSession!!)
        } catch (e: Exception) {
            Toast.makeText(this, "ARCore not supported", Toast.LENGTH_LONG).show()
            finish()
        }

        // Handle Taps for Anchor Placement (Delegated to Renderer for Thread-Safety)
        glSurfaceView.setOnTouchListener { _, event ->
            if (event.action == android.view.MotionEvent.ACTION_UP) {
                renderer.onTouch(event)
            }
            true
        }
    }

    fun onAnchorPlaced() {
        binding.statusText.text = "Box anchored. Adjust fit below."
    }

    fun onFrameRendered(frame: Frame, mvp: FloatArray, model: FloatArray, view: FloatArray) {
        if (isRecording && !isProcessingFrame) {
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastFrameTime >= 33) { // ~30 FPS
                processFrameForRecording(frame, model, view, mvp)
                lastFrameTime = currentTime
            }
        }
        
        // Update UI status
        runOnUiThread {
            if (renderer.currentAnchor == null) {
                binding.statusText.text = "Scanning for surfaces... Tap to place."
            }
        }
    }

    private fun processFrameForRecording(frame: Frame, model: FloatArray, view: FloatArray, mvp: FloatArray) {
        isProcessingFrame = true
        val camera = frame.camera
        
        // OBJECTRON PERFECT SYNC: Use direct sensor view matrix
        val sensorViewMatrix = FloatArray(16)
        camera.pose.inverse().toMatrix(sensorViewMatrix, 0)

        // Extract Sparse Point Cloud
        val pointCloud = mutableListOf<List<Float>>()
        try {
            val rawCloud = frame.acquirePointCloud()
            val buffer = rawCloud.points
            val count = buffer.remaining() / 4
            for (i in 0 until count) {
                pointCloud.add(listOf(buffer.get(i * 4), buffer.get(i * 4 + 1), buffer.get(i * 4 + 2)))
            }
            rawCloud.release()
        } catch (e: Exception) {}

        val intrinsics = camera.imageIntrinsics
        val frameId = frameCount++
        val imageName = "frame_${String.format("%04d", frameId)}.jpg"
        
        val entry = AnnotationGenerator.createEntry(
            frameId, imageName, model, sensorViewMatrix, mvp, pointCloud,
            intrinsics.focalLength[0], intrinsics.focalLength[1],
            intrinsics.principalPoint[0], intrinsics.principalPoint[1],
            intrinsics.imageDimensions[0], intrinsics.imageDimensions[1],
            System.currentTimeMillis()
        )

        // CAPTURE RAW FRAME WITH OVERLAY
        // We capture binding.root or glSurfaceView to ensure annotations are saved in the frame as requested.
        val bitmap = Bitmap.createBitmap(glSurfaceView.width, glSurfaceView.height, Bitmap.Config.ARGB_8888)
        PixelCopy.request(glSurfaceView, bitmap, { result ->
            if (result == PixelCopy.SUCCESS) {
                captureManager.saveFrame(bitmap, entry)
            }
            isProcessingFrame = false
        }, glSurfaceView.handler)
    }

    override fun onResume() {
        super.onResume()
        arSession?.resume()
        glSurfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        arSession?.pause()
        glSurfaceView.onPause()
    }

    private fun setupControls() {
        setupRow(binding.ctrlScaleX, "Scale X", scaleX, 0.5f) { scaleX = it; renderer.mScaleX = it }
        setupRow(binding.ctrlScaleY, "Scale Y", scaleY, 0.5f) { scaleY = it; renderer.mScaleY = it }
        setupRow(binding.ctrlScaleZ, "Scale Z", scaleZ, 0.5f) { scaleZ = it; renderer.mScaleZ = it }

        setupRow(binding.ctrlRotY, "Rot Y", rotY, 5f) { rotY = it; renderer.mRotationY = it }

        setupRow(binding.ctrlTransX, "Trans X", transX, 1f) { transX = it; renderer.mTranslationX = it }
        setupRow(binding.ctrlTransY, "Trans Y", transY, 1f) { transY = it; renderer.mTranslationY = it }
        setupRow(binding.ctrlTransZ, "Trans Z", transZ, 1f) { transZ = it; renderer.mTranslationZ = it }

        binding.btnRecord.setOnClickListener { toggleRecording() }
        binding.btnExport.setOnClickListener {
            val capturePath = captureManager.getCapturePath()
            val directory = File(capturePath)
            if (directory.exists() && !isRecording) {
                val zipFile = File(directory.parent, "${directory.name}.zip")
                ZipUtils.zipDirectory(directory, zipFile)
                Toast.makeText(this, "Dataset zipped to: ${zipFile.absolutePath}", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "Stop recording first or capture data.", Toast.LENGTH_SHORT).show()
            }
        }
        binding.btnReset.setOnClickListener {
            resetTransforms()
        }
    }

    private fun resetTransforms() {
        scaleX = 6.5f; scaleY = 12f; scaleZ = 6.5f
        rotY = 0f
        transX = 0f; transY = 6f; transZ = 0f
        
        renderer.mScaleX = scaleX; renderer.mScaleY = scaleY; renderer.mScaleZ = scaleZ
        renderer.mRotationY = rotY
        renderer.mTranslationX = transX; renderer.mTranslationY = transY; renderer.mTranslationZ = transZ
        
        renderer.resetAnchor()
        
        binding.statusText.text = "Transforms reset. Tap to re-anchor."
        Toast.makeText(this, "All values reset to defaults", Toast.LENGTH_SHORT).show()
    }

    private fun setupRow(rowBinding: com.example.arcoreapp.databinding.LayoutControlRowBinding, label: String, initialValue: Float, step: Float = 0.5f, onUpdate: (Float) -> Unit) {
        var current = initialValue
        rowBinding.label.text = label
        rowBinding.valueText.text = String.format("%.1f", current)
        rowBinding.btnPlus.setOnClickListener {
            current += step
            rowBinding.valueText.text = String.format("%.1f", current)
            onUpdate(current)
        }
        rowBinding.btnMinus.setOnClickListener {
            current -= step
            if (label.contains("Scale") && current < 0.1f) current = 0.1f
            rowBinding.valueText.text = String.format("%.1f", current)
            onUpdate(current)
        }
    }

    private fun toggleRecording() {
        if (!isRecording) {
            if (renderer.currentAnchor == null) {
                Toast.makeText(this, "Place box first!", Toast.LENGTH_SHORT).show()
                return
            }
            isRecording = true
            frameCount = 0
            captureManager.startNewSequence()
            binding.btnRecord.text = "STOP RECORDING"
            binding.fpsText.text = "Status: RECORDING..."
        } else {
            isRecording = false
            captureManager.finishSequence()
            binding.btnRecord.text = "START RECORDING"
            binding.fpsText.text = "Status: IDLE (Saved ${frameCount} frames)"
        }
    }
}
