package com.example.arcoreapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.opengl.Matrix
import android.os.Bundle
import android.view.PixelCopy
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.arcoreapp.databinding.ActivityMainBinding
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ArSceneView
import io.github.sceneview.ar.node.ArNode
import io.github.sceneview.math.Rotation
import io.github.sceneview.math.Scale
import io.github.sceneview.math.Position
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var sceneView: ArSceneView
    private var boxNode: ArNode? = null // This will be the anchored parent
    private var transformableNode: ArNode? = null // This will be the child we transform
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
    private var transY = 0.06f // Center the box so bottom is on plane (0.12 / 2)
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

        sceneView = binding.sceneView
        captureManager = CaptureManager(this)

        if (checkCameraPermission()) {
            setupScene()
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
                setupScene()
            } else {
                Toast.makeText(this, "Camera permission is required for AR", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun setupScene() {
        sceneView.apply {
            planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
            focusMode = Config.FocusMode.AUTO
            lightEstimationMode = Config.LightEstimationMode.ENVIRONMENTAL_HDR
            
            // Configure session for depth and stability
            onSessionConfiguration = { session, config ->
                if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    config.depthMode = Config.DepthMode.AUTOMATIC
                }
                config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            }

            planeRenderer.isVisible = false // Hide the dotted patterns
            
            onArFrame = { frame ->
                val arFrame = frame.frame
                val planes = arFrame.getUpdatedTrackables(Plane::class.java).filter { it.trackingState == TrackingState.TRACKING }
                if (planes.isNotEmpty() && boxNode == null) {
                    binding.statusText.text = "Plane detected. Tap to place box."
                }
                
                updateOverlay(frame)
                if (isRecording && !isProcessingFrame) {
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastFrameTime >= 33) { // ~30 FPS
                        processFrameForRecording(frame)
                        lastFrameTime = currentTime
                    }
                }
            }

            onTapAr = { hitResult, _ ->
                if (boxNode == null) {
                    // Create an anchor at the hit location
                    val anchor = hitResult.createAnchor()
                    placeBox(anchor)
                }
            }
        }
    }

    private fun placeBox(anchor: Anchor) {
        val node = ArNode(sceneView.engine)
        node.anchor = anchor
        sceneView.addChild(node)
        boxNode = node
        
        // We still create a child node for visual representation if needed, 
        // but our overlay logic uses calculateModelMatrix(anchor).
        val childNode = ArNode(sceneView.engine)
        node.addChild(childNode)
        transformableNode = childNode

        updateBoxTransform()
        binding.statusText.text = "Box placed. Use buttons to fit the can."
    }

    private fun updateOverlay(frame: io.github.sceneview.ar.arcore.ArFrame) {
        val anchor = boxNode?.anchor ?: return
        if (anchor.trackingState != TrackingState.TRACKING) return

        val camera = frame.camera
        val viewMatrix = FloatArray(16)
        camera.getViewMatrix(viewMatrix, 0)
        
        // Calculate the model matrix manually for maximum precision and reactivity
        val modelMatrix = calculateModelMatrix(anchor)

        val intrinsics = camera.imageIntrinsics
        val fx = intrinsics.focalLength[0]
        val fy = intrinsics.focalLength[1]
        val cx = intrinsics.principalPoint[0]
        val cy = intrinsics.principalPoint[1]
        val width = intrinsics.imageDimensions[0]
        val height = intrinsics.imageDimensions[1]

        val entry = AnnotationGenerator.createEntry(
            0, "", modelMatrix, viewMatrix,
            fx, fy, cx, cy, width, height, 0
        )
        
        runOnUiThread {
            binding.boxOverlay.updatePoints(entry.keypoints2d)
        }
    }

    private fun calculateModelMatrix(anchor: Anchor): FloatArray {
        // 1. Get Anchor Pose
        val anchorPose = anchor.pose
        
        // 2. Create local transformation pose (Translation + Rotation)
        // Convert Euler angles to Quaternion
        val quat = eulerToQuaternion(rotX, rotY, rotZ)
        val localPose = Pose.makeTranslation(transX, transY, transZ)
            .compose(Pose.makeRotation(quat[0], quat[1], quat[2], quat[3]))
        
        // 3. Combine to get World Pose
        val worldPose = anchorPose.compose(localPose)
        
        // 4. Convert to Matrix and apply Scale
        val matrix = FloatArray(16)
        worldPose.toMatrix(matrix, 0)
        android.opengl.Matrix.scaleM(matrix, 0, scaleX, scaleY, scaleZ)
        
        return matrix
    }

    private fun eulerToQuaternion(pitch: Float, yaw: Float, roll: Float): FloatArray {
        val p = Math.toRadians(pitch.toDouble()).toFloat()
        val y = Math.toRadians(yaw.toDouble()).toFloat()
        val r = Math.toRadians(roll.toDouble()).toFloat()

        val c1 = Math.cos((y / 2).toDouble()).toFloat()
        val s1 = Math.sin((y / 2).toDouble()).toFloat()
        val c2 = Math.cos((p / 2).toDouble()).toFloat()
        val s2 = Math.sin((p / 2).toDouble()).toFloat()
        val c3 = Math.cos((r / 2).toDouble()).toFloat()
        val s3 = Math.sin((r / 2).toDouble()).toFloat()

        return floatArrayOf(
            s1 * s2 * c3 + c1 * c2 * s3, // x
            s1 * c2 * c3 + c1 * s2 * s3, // y
            c1 * s2 * c3 - s1 * c2 * s3, // z
            c1 * c2 * c3 - s1 * s2 * s3  // w
        )
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
        val anchor = boxNode?.anchor ?: return
        isProcessingFrame = true
        val camera = frame.camera
        val viewMatrix = FloatArray(16)
        camera.getViewMatrix(viewMatrix, 0)
        
        // Use the same robust matrix calculation as the overlay
        val modelMatrix = calculateModelMatrix(anchor)

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
            frameId, imageName, modelMatrix, viewMatrix,
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
        // We still update the transformableNode so it renders correctly in 3D space 
        // if there's any model attached, but our primary tracking is now via calculateModelMatrix.
        transformableNode?.let { node ->
            node.scale = Scale(scaleX, scaleY, scaleZ)
            node.rotation = Rotation(rotX, rotY, rotZ)
            node.position = Position(transX, transY, transZ)
        }
    }
}
