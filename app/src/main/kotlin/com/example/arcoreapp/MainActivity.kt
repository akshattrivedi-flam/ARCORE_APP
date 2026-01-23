package com.example.arcoreapp

import android.graphics.Bitmap
import android.os.Bundle
import android.view.PixelCopy
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.arcoreapp.databinding.ActivityMainBinding
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import io.github.sceneview.ar.ArSceneView
import io.github.sceneview.ar.node.ArNode
import io.github.sceneview.math.Rotation
import io.github.sceneview.math.Scale
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var sceneView: ArSceneView
    private var boxNode: ArNode? = null
    private lateinit var captureManager: CaptureManager
    private var frameCount = 0

    // Box properties (Manual fitting)
    private var scaleX = 0.065f
    private var scaleY = 0.12f
    private var scaleZ = 0.065f
    private var rotX = 0f
    private var rotY = 0f
    private var rotZ = 0f
    private var transX = 0f
    private var transY = 0f
    private var transZ = 0f

    private var isRecording = false
    private var isProcessingFrame = false
    private var lastFrameTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        sceneView = binding.sceneView
        captureManager = CaptureManager(this)

        setupScene()
        setupControls()
    }

    private fun setupScene() {
        sceneView.apply {
            lightEstimationMode = Config.LightEstimationMode.DISABLED
            planeRenderer.isVisible = true
            
            onArFrame = { frame ->
                updateOverlay(frame)
                if (isRecording && !isProcessingFrame) {
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastFrameTime >= 16) { 
                        processFrameForRecording(frame)
                        lastFrameTime = currentTime
                    }
                }
            }

            onTapAr = { hitResult, _ ->
                if (boxNode == null) {
                    placeBox(hitResult.createAnchor())
                }
            }
        }
    }

    private fun placeBox(anchor: Anchor) {
        val node = ArNode(sceneView.engine)
        node.anchor = anchor
        sceneView.addChild(node)
        boxNode = node
        updateBoxTransform()
        binding.statusText.text = "Box placed. Use buttons to fit the can."
    }

    private fun updateOverlay(frame: io.github.sceneview.ar.arcore.ArFrame) {
        val node = boxNode ?: return
        val camera = frame.camera
        val viewMatrix = FloatArray(16)
        camera.getViewMatrix(viewMatrix, 0)
        
        val intrinsics = camera.imageIntrinsics
        val fx = intrinsics.focalLength[0]
        val fy = intrinsics.focalLength[1]
        val cx = intrinsics.principalPoint[0]
        val cy = intrinsics.principalPoint[1]
        val width = intrinsics.imageDimensions[0]
        val height = intrinsics.imageDimensions[1]

        val entry = AnnotationGenerator.createEntry(
            0, "", node.worldTransform.toFloatArray(), viewMatrix,
            fx, fy, cx, cy, width, height, 0
        )
        
        runOnUiThread {
            binding.boxOverlay.updatePoints(entry.keypoints2d)
        }
    }

    private fun setupControls() {
        setupRow(binding.ctrlScaleX, "Scale X", scaleX) { scaleX = it; updateBoxTransform() }
        setupRow(binding.ctrlScaleY, "Scale Y", scaleY) { scaleY = it; updateBoxTransform() }
        setupRow(binding.ctrlScaleZ, "Scale Z", scaleZ) { scaleZ = it; updateBoxTransform() }

        setupRow(binding.ctrlRotX, "Rot X", rotX, 5f) { rotX = it; updateBoxTransform() }
        setupRow(binding.ctrlRotY, "Rot Y", rotY, 5f) { rotY = it; updateBoxTransform() }
        setupRow(binding.ctrlRotZ, "Rot Z", rotZ, 5f) { rotZ = it; updateBoxTransform() }

        setupRow(binding.ctrlTransX, "Trans X", transX, 0.01f) { transX = it; updateBoxTransform() }
        setupRow(binding.ctrlTransY, "Trans Y", transY, 0.01f) { transY = it; updateBoxTransform() }
        setupRow(binding.ctrlTransZ, "Trans Z", transZ, 0.01f) { transZ = it; updateBoxTransform() }

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
    }

    private fun setupRow(rowBinding: com.example.arcoreapp.databinding.LayoutControlRowBinding, label: String, initialValue: Float, step: Float = 0.005f, onUpdate: (Float) -> Unit) {
        var current = initialValue
        rowBinding.label.text = label
        rowBinding.valueText.text = String.format("%.3f", current)
        rowBinding.btnPlus.setOnClickListener {
            current += step
            rowBinding.valueText.text = String.format("%.3f", current)
            onUpdate(current)
        }
        rowBinding.btnMinus.setOnClickListener {
            current -= step
            if (label.contains("Scale") && current < 0.001f) current = 0.001f
            rowBinding.valueText.text = String.format("%.3f", current)
            onUpdate(current)
        }
    }

    private fun toggleRecording() {
        if (!isRecording) {
            if (boxNode == null) {
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

    private fun processFrameForRecording(frame: io.github.sceneview.ar.arcore.ArFrame) {
        val node = boxNode ?: return
        isProcessingFrame = true
        val camera = frame.camera
        val viewMatrix = FloatArray(16)
        camera.getViewMatrix(viewMatrix, 0)
        
        val intrinsics = camera.imageIntrinsics
        val fx = intrinsics.focalLength[0]
        val fy = intrinsics.focalLength[1]
        val cx = intrinsics.principalPoint[0]
        val cy = intrinsics.principalPoint[1]
        val width = intrinsics.imageDimensions[0]
        val height = intrinsics.imageDimensions[1]

        val frameId = frameCount++
        val imageName = "frame_${String.format("%04d", frameId)}.jpg"
        val entry = AnnotationGenerator.createEntry(
            frameId, imageName, node.worldTransform.toFloatArray(), viewMatrix,
            fx, fy, cx, cy, width, height, System.currentTimeMillis()
        )

        val bitmap = Bitmap.createBitmap(sceneView.width, sceneView.height, Bitmap.Config.ARGB_8888)
        PixelCopy.request(sceneView, bitmap, { result ->
            if (result == PixelCopy.SUCCESS) {
                captureManager.saveFrame(bitmap, entry)
            }
            isProcessingFrame = false
        }, sceneView.handler)
    }

    private fun updateBoxTransform() {
        boxNode?.let { node ->
            node.scale = Scale(scaleX, scaleY, scaleZ)
            node.rotation = Rotation(rotX, rotY, rotZ)
            node.position = io.github.sceneview.math.Position(transX, transY, transZ)
        }
    }
}
