package com.example.arcoreapp

import android.graphics.Bitmap
import android.os.Bundle
import android.view.PixelCopy
import android.view.View
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.arcoreapp.databinding.ActivityMainBinding
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import com.google.ar.core.Plane
import io.github.sceneview.ar.ArSceneView
import io.github.sceneview.ar.node.ArNode
import io.github.sceneview.math.Rotation
import io.github.sceneview.math.Scale
import io.github.sceneview.node.CubeNode
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var sceneView: ArSceneView
    private var boxNode: ArNode? = null
    private lateinit var captureManager: CaptureManager
    private var frameCount = 0

    // Box properties (Manual fitting)
    private var scaleX = 0.065f // Standard can diameter ~6.5cm
    private var scaleY = 0.12f   // Standard can height ~12cm
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
                if (isRecording && !isProcessingFrame) {
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastFrameTime >= 16) { 
                        processFrameForRecording(frame)
                        lastFrameTime = currentTime
                    }
                }
            }

            onTapAr = { hitResult ->
                if (boxNode == null) {
                    placeBox(hitResult.createAnchor())
                }
            }
        }
    }

    private fun placeBox(anchor: Anchor) {
        val node = ArNode(sceneView.engine)
        node.anchor = anchor
        
        val cube = CubeNode(sceneView.engine, size = Scale(1f, 1f, 1f))
        node.addChild(cube)
        
        sceneView.addChild(node)
        boxNode = node
        updateBoxTransform()
        
        binding.statusText.text = "Box placed. Use buttons to fit the can."
    }

    private fun setupControls() {
        // SCALE
        setupRow(binding.ctrlScaleX, "Scale X", scaleX) { scaleX = it; updateBoxTransform() }
        setupRow(binding.ctrlScaleY, "Scale Y", scaleY) { scaleY = it; updateBoxTransform() }
        setupRow(binding.ctrlScaleZ, "Scale Z", scaleZ) { scaleZ = it; updateBoxTransform() }

        // ROTATION
        setupRow(binding.ctrlRotX, "Rot X", rotX, 5f) { rotX = it; updateBoxTransform() }
        setupRow(binding.ctrlRotY, "Rot Y", rotY, 5f) { rotY = it; updateBoxTransform() }
        setupRow(binding.ctrlRotZ, "Rot Z", rotZ, 5f) { rotZ = it; updateBoxTransform() }

        // TRANSLATION
        setupRow(binding.ctrlTransX, "Trans X", transX, 0.01f) { transX = it; updateBoxTransform() }
        setupRow(binding.ctrlTransY, "Trans Y", transY, 0.01f) { transY = it; updateBoxTransform() }
        setupRow(binding.ctrlTransZ, "Trans Z", transZ, 0.01f) { transZ = it; updateBoxTransform() }

        binding.btnRecord.setOnClickListener {
            toggleRecording()
        }

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

    private fun updateBoxTransform() {
        boxNode?.let { node ->
            node.scale = Scale(scaleX, scaleY, scaleZ)
            node.rotation = Rotation(rotX, rotY, rotZ)
            node.position = io.github.sceneview.math.Position(transX, transY, transZ)
        }
    }
}
