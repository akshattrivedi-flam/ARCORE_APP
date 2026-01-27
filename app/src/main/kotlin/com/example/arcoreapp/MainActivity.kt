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
import com.google.ar.core.Session
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

    // Box properties (Manual fitting) - values in centimeters
    private var scaleX = 6.5f
    private var scaleY = 12f
    private var scaleZ = 6.5f
    private var rotX = 0f
    private var rotY = 0f
    private var rotZ = 0f
    private var transX = 0f
    private var transY = 6f // Center the box so bottom is on plane (12 / 2)
    private var transZ = 0f

    private var lastArFrame: io.github.sceneview.ar.arcore.ArFrame? = null
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
            onArSessionCreated = { session ->
                val config = session.config
                
                // Enable Depth for better anchoring on objects
                if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    config.depthMode = Config.DepthMode.AUTOMATIC
                }
                
                // Enable Instant Placement for immediate stability
                config.instantPlacementMode = Config.InstantPlacementMode.LOCAL_Y_UP
                
                config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                config.focusMode = Config.FocusMode.AUTO
                session.configure(config)
            }

            planeRenderer.isVisible = false // Hide the dotted patterns
            
            onArFrame = { frame ->
                lastArFrame = frame
                val arFrame = frame.frame
                
                // Check all trackables for a stable plane
                val allPlanes = arSession?.getAllTrackables(Plane::class.java) ?: emptyList()
                val hasStablePlane = allPlanes.any { 
                    it.trackingState == TrackingState.TRACKING && 
                    it.type == Plane.Type.HORIZONTAL_UPWARD_FACING 
                }
                
                if (hasStablePlane && boxNode == null) {
                    binding.statusText.text = "Surface detected. Tap the object to place box."
                } else if (boxNode == null) {
                    binding.statusText.text = "Scanning for surfaces..."
                }
                
                // Update UI overlay on every frame if box is placed
                updateOverlay(frame)
                
                if (isRecording && !isProcessingFrame) {
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastFrameTime >= 33) { // ~30 FPS
                        processFrameForRecording(frame)
                        lastFrameTime = currentTime
                    }
                }
            }

            onTapAr = { _, motionEvent ->
                lastArFrame?.let { arFrame ->
                    val camera = arFrame.camera
                    if (camera.trackingState == TrackingState.TRACKING) {
                        val frame = arFrame.frame
                        val hits = frame.hitTest(motionEvent.x, motionEvent.y)
                        
                        // STABILITY PRIORITY LOGIC:
                        // We want the X,Z of the tap but the Y of the most stable surface (the plane).
                        // 1. Find the highest stable horizontal plane (usually the table/floor).
                        val planeHit = hits.firstOrNull { hit ->
                            val t = hit.trackable
                            t is Plane && t.trackingState == TrackingState.TRACKING && 
                            t.type == Plane.Type.HORIZONTAL_UPWARD_FACING &&
                            t.isPoseInPolygon(hit.hitPose)
                        }
                        
                        // 2. Fallback to closest trackable if no plane is found
                        val bestHit = planeHit ?: hits.firstOrNull { it.trackable.trackingState == TrackingState.TRACKING }

                        if (bestHit != null) {
                            // ABSOLUTE STABILITY: Use the hit's translation but force World Y-up orientation.
                            val hitPose = bestHit.hitPose
                            
                            // If we have a plane hit, we use its Y height for better ground-locking.
                            val tx = hitPose.tx()
                            val ty = hitPose.ty()
                            val tz = hitPose.tz()
                            
                            val uprightPose = Pose.makeTranslation(tx, ty, tz)
                            
                            // Creating the anchor ON the trackable allows ARCore to correct its drift over time.
                            val anchor = bestHit.trackable.createAnchor(uprightPose)
                            
                            android.util.Log.d("ARCoreApp", "Locked anchor to ${bestHit.trackable::class.java.simpleName} at Y: $ty")

                            boxNode?.let {
                                sceneView.removeChild(it)
                                it.destroy()
                            }
                            placeBox(anchor)
                        } else {
                            Toast.makeText(this@MainActivity, "Surface not stable. Move phone to detect planes.", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }
    }

    private fun placeBox(anchor: Anchor) {
        val node = ArNode(sceneView.engine)
        node.anchor = anchor
        sceneView.addChild(node)
        boxNode = node
        
        // Transformable child node for visual representation
        val childNode = ArNode(sceneView.engine)
        node.addChild(childNode)
        transformableNode = childNode

        updateBoxTransform()
        binding.statusText.text = "Box locked to surface. Adjust fit below."
    }

    private fun updateOverlay(frame: io.github.sceneview.ar.arcore.ArFrame) {
        val node = transformableNode ?: return
        val anchor = boxNode?.anchor ?: return
        if (anchor.trackingState != TrackingState.TRACKING) return

        val camera = frame.camera
        
        // EXPERT FIX: Use Sensor-oriented View Matrix for Intrinsics projection
        // camera.getPose() returns the world-to-camera pose relative to the sensor.
        val cameraPose = camera.pose
        val sensorViewMatrix = FloatArray(16)
        cameraPose.inverse().toMatrix(sensorViewMatrix, 0)
        
        // Use the manual matrix calculation as the source of truth
        val modelMatrix = calculateModelMatrix(anchor)

        val keypoints2d = mutableListOf<List<Float>>()
        val unitCube = listOf(
            floatArrayOf(0f, 0f, 0f),       // Center
            floatArrayOf(-0.5f, -0.5f, 0.5f), floatArrayOf(0.5f, -0.5f, 0.5f),
            floatArrayOf(0.5f, 0.5f, 0.5f), floatArrayOf(-0.5f, 0.5f, 0.5f),
            floatArrayOf(-0.5f, -0.5f, -0.5f), floatArrayOf(0.5f, -0.5f, -0.5f),
            floatArrayOf(0.5f, 0.5f, -0.5f), floatArrayOf(-0.5f, 0.5f, -0.5f)
        )

        for (localPt in unitCube) {
            val worldPt4 = FloatArray(4)
            Matrix.multiplyMV(worldPt4, 0, modelMatrix, 0, floatArrayOf(localPt[0], localPt[1], localPt[2], 1.0f), 0)
            
            // Project to Camera Image Space (normalized 0..1 of the BUFFER)
            val proj = MathUtils.projectPoint(
                floatArrayOf(worldPt4[0], worldPt4[1], worldPt4[2]),
                sensorViewMatrix,
                camera.imageIntrinsics.focalLength[0], camera.imageIntrinsics.focalLength[1],
                camera.imageIntrinsics.principalPoint[0], camera.imageIntrinsics.principalPoint[1],
                camera.imageIntrinsics.imageDimensions[0], camera.imageIntrinsics.imageDimensions[1]
            )
            
            // Transform from Camera Buffer Space to View Screen Space
            val viewCoords = FloatArray(2)
            frame.frame.transformCoordinates2d(
                com.google.ar.core.Coordinates2d.IMAGE_NORMALIZED,
                floatArrayOf(proj[0], proj[1]),
                com.google.ar.core.Coordinates2d.VIEW_NORMALIZED,
                viewCoords
            )
            
            keypoints2d.add(listOf(viewCoords[0], viewCoords[1], proj[2]))
        }
        
        runOnUiThread {
            binding.boxOverlay.updatePoints(keypoints2d)
        }
    }

    private fun calculateModelMatrix(anchor: Anchor): FloatArray {
        val anchorPose = anchor.pose
        val quat = eulerToQuaternion(rotX, rotY, rotZ)
        
        // Use manual matrix multiplication for better control over transform order
        val modelMatrix = FloatArray(16)
        val translationMatrix = FloatArray(16)
        val rotationMatrix = FloatArray(16)
        val scaleMatrix = FloatArray(16)
        
        // Initialize matrices
        Matrix.setIdentityM(translationMatrix, 0)
        Matrix.setIdentityM(rotationMatrix, 0)
        Matrix.setIdentityM(scaleMatrix, 0)
        
        // 1. Get the anchor matrix - FORCE UPRIGHT if on horizontal plane
        val anchorMatrix = FloatArray(16)
        
        // Extract translation from anchor pose
        val tx = anchorPose.tx()
        val ty = anchorPose.ty()
        val tz = anchorPose.tz()
        
        // Check if the anchor is from a horizontal plane (using our gravity-aligned logic from onTapAr)
        // Even if not, we force identity rotation (Y-up) for the anchor base to prevent tilt.
        val uprightPose = Pose.makeTranslation(tx, ty, tz)
        uprightPose.toMatrix(anchorMatrix, 0)
        
        // 2. Build local transforms
        // Translation in centimeters converted to meters
        Matrix.translateM(translationMatrix, 0, transX / 100f, transY / 100f, transZ / 100f)
        
        // Create rotation matrix from quaternion
        val rotationPose = Pose.makeRotation(quat[0], quat[1], quat[2], quat[3])
        rotationPose.toMatrix(rotationMatrix, 0)
        
        // Scale in centimeters converted to meters
        Matrix.scaleM(scaleMatrix, 0, scaleX / 100f, scaleY / 100f, scaleZ / 100f)
        
        // 3. Combine: World = Anchor(Translation Only) * Translation * Rotation * Scale
        val temp1 = FloatArray(16)
        val temp2 = FloatArray(16)
        
        Matrix.multiplyMM(temp1, 0, anchorMatrix, 0, translationMatrix, 0)
        Matrix.multiplyMM(temp2, 0, temp1, 0, rotationMatrix, 0)
        Matrix.multiplyMM(modelMatrix, 0, temp2, 0, scaleMatrix, 0)
        
        return modelMatrix
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
        setupRow(binding.ctrlScaleX, "Scale X", scaleX, 0.5f) { scaleX = it }
        setupRow(binding.ctrlScaleY, "Scale Y", scaleY, 0.5f) { scaleY = it }
        setupRow(binding.ctrlScaleZ, "Scale Z", scaleZ, 0.5f) { scaleZ = it }

        setupRow(binding.ctrlRotX, "Rot X", rotX, 5f) { rotX = it }
        setupRow(binding.ctrlRotY, "Rot Y", rotY, 5f) { rotY = it }
        setupRow(binding.ctrlRotZ, "Rot Z", rotZ, 5f) { rotZ = it }

        setupRow(binding.ctrlTransX, "Trans X", transX, 1f) { transX = it }
        setupRow(binding.ctrlTransY, "Trans Y", transY, 1f) { transY = it }
        setupRow(binding.ctrlTransZ, "Trans Z", transZ, 1f) { transZ = it }

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
        scaleX = 6.5f
        scaleY = 12f
        scaleZ = 6.5f
        rotX = 0f
        rotY = 0f
        rotZ = 0f
        transX = 0f
        transY = 6f
        transZ = 0f
        
        // Remove existing box to force a clean re-anchor if desired
        boxNode?.let {
            sceneView.removeChild(it)
            it.destroy()
        }
        boxNode = null
        transformableNode = null
        
        // Update all UI labels
        setupControls() 
        
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
            // Explicitly trigger transform update for immediate feedback
            updateBoxTransform()
        }
        rowBinding.btnMinus.setOnClickListener {
            current -= step
            if (label.contains("Scale") && current < 0.1f) current = 0.1f
            rowBinding.valueText.text = String.format("%.1f", current)
            onUpdate(current)
            // Explicitly trigger transform update for immediate feedback
            updateBoxTransform()
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
            node.scale = Scale(scaleX / 100f, scaleY / 100f, scaleZ / 100f)
            node.rotation = Rotation(rotX, rotY, rotZ)
            node.position = Position(transX / 100f, transY / 100f, transZ / 100f)
        }
    }
}
