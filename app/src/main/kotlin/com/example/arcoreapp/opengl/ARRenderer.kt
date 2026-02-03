package com.example.arcoreapp.opengl

import android.content.Context
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import com.google.ar.core.Anchor
import com.google.ar.core.Session
import com.example.arcoreapp.MainActivity
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class ARRenderer(private val context: Context) : GLSurfaceView.Renderer {

    private var backgroundRenderer = BackgroundRenderer()
    private var customBox: CustomBoundingBox? = null
    private var customCylinder: CustomCylinder? = null
    private var program = -1
    private var session: Session? = null
    
    private var mvpMatrix = FloatArray(16)
    private var projectionMatrix = FloatArray(16)
    private var viewMatrix = FloatArray(16)
    private var anchorMatrix = FloatArray(16)
    private var smoothedMatrix = FloatArray(16).apply { Matrix.setIdentityM(this, 0) }
    private val SMOOTHING_FACTOR = 0.15f // Lower = smoother, higher = faster

    private var positionHandle = -1
    private var colorHandle = -1
    private var mvpMatrixHandle = -1
    private var tintHandle = -1

    private val vertexShaderCode = """
        uniform mat4 uMVPMatrix;
        attribute vec4 vPosition;
        attribute vec4 aColor;
        varying vec4 vColor;
        void main() {
          gl_Position = uMVPMatrix * vPosition;
          vColor = aColor; 
        }
    """.trimIndent()

    private val fragmentShaderCode = """
        precision mediump float;
        uniform vec4 uTint;
        varying vec4 vColor;
        void main() {
          gl_FragColor = vColor * uTint;
        }
    """.trimIndent()

    // Transform properties (centimeters) - MARKED VOLATILE FOR THREAD SAFETY
    @Volatile var mScaleX = 7.0f
    @Volatile var mScaleY = 15.0f
    @Volatile var mScaleZ = 7.0f
    @Volatile var mRotationX = 0.0f
    @Volatile var mRotationY = 0.0f
    @Volatile var mRotationZ = 0.0f
    @Volatile var mTranslationX = 0.0f
    @Volatile var mTranslationY = 0.0f
    @Volatile var mTranslationZ = 0.0f // This will now act as our "Depth" control
    @Volatile var mManualDepth = 50.0f // Initial placement distance in cm
    @Volatile var isCameraLocked = false // Handheld stability mode
    @Volatile var isAutoDepthEnabled = false // Continuous depth tracking
    @Volatile var isCylinderMode = false // Toggle between Box and Cylinder

    // Bounding Box Color [R, G, B, A]
    @Volatile var mBoxColor = floatArrayOf(1.0f, 0.0f, 0.0f, 0.3f) // Default Red

    fun updateBoxColor(color: FloatArray) {
        mBoxColor = color
    }

    @Volatile var currentAnchor: Anchor? = null
        private set
    
    @Volatile var trackedImage: com.google.ar.core.AugmentedImage? = null

    @Synchronized
    fun resetAnchor() {
        currentAnchor?.detach()
        currentAnchor = null
    }

    fun placeInAir(useDepth: Boolean = true) {
        val session = this.session ?: return
        try {
            val frame = session.update()
            val cameraPose = frame.camera.pose
            
            var distanceInMeters = mManualDepth / 100f
            
            if (useDepth) {
                // Better Depth API usage: Sample center of screen
                val hits = frame.hitTest(frame.camera.imageIntrinsics.imageDimensions[0] / 2f, 
                                        frame.camera.imageIntrinsics.imageDimensions[1] / 2f)
                val depthHit = hits.firstOrNull { it.trackable is com.google.ar.core.DepthPoint }
                if (depthHit != null) {
                    distanceInMeters = depthHit.hitPose.tz().let { Math.abs(it) } 
                    mManualDepth = distanceInMeters * 100f
                }
            }

            val relativePose = com.google.ar.core.Pose.makeTranslation(0f, 0f, -distanceInMeters)
            val airPose = cameraPose.compose(relativePose)
            
            resetAnchor()
            synchronized(this) {
                currentAnchor = session.createAnchor(airPose)
            }
            
            (context as MainActivity).runOnUiThread {
                context.onAnchorPlaced()
            }
        } catch (e: Exception) {}
    }

    private val queuedTaps = java.util.concurrent.ArrayBlockingQueue<android.view.MotionEvent>(16)

    fun onTouch(event: android.view.MotionEvent) {
        queuedTaps.offer(event)
    }

    fun setArSession(session: Session) {
        this.session = session
    }

    fun snapToImage(image: com.google.ar.core.AugmentedImage) {
        val session = this.session ?: return
        // Create an anchor at the center of the image
        resetAnchor()
        synchronized(this) {
            currentAnchor = image.createAnchor(image.centerPose)
        }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.1f, 0.1f, 0.1f, 1.0f)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)

        backgroundRenderer.createOnGlThread()
        customBox = CustomBoundingBox()
        customCylinder = CustomCylinder(50)

        val vertexShader = loadShader(GLES30.GL_VERTEX_SHADER, vertexShaderCode)
        val fragmentShader = loadShader(GLES30.GL_FRAGMENT_SHADER, fragmentShaderCode)

        program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        val displayRotation = (context as MainActivity).windowManager.defaultDisplay.rotation
        session?.setDisplayGeometry(displayRotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)

        val session = this.session ?: return
        
        try {
            session.setCameraTextureName(backgroundRenderer.textureId)
            val frame = session.update()
            val camera = frame.camera

            backgroundRenderer.draw(frame)
            
            // Continuous Depth Sync logic
            if (isAutoDepthEnabled) {
                try {
                    val hits = frame.hitTest(frame.camera.imageIntrinsics.imageDimensions[0] / 2f, 
                                            frame.camera.imageIntrinsics.imageDimensions[1] / 2f)
                    val depthHit = hits.firstOrNull { it.trackable is com.google.ar.core.DepthPoint }
                    if (depthHit != null) {
                        val distanceInMeters = depthHit.hitPose.tz().let { Math.abs(it) }
                        mManualDepth = distanceInMeters * 100f
                    }
                } catch (e: Exception) {}
            }

            handleTaps(frame, camera)

            if (camera.trackingState == com.google.ar.core.TrackingState.TRACKING) {
                camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100.0f)
                camera.getViewMatrix(viewMatrix, 0)

                // --- DYNAMIC TARGET LOGIC (Follow-the-Marker) ---
                val activeImage = trackedImage
                if (activeImage != null && activeImage.trackingState == com.google.ar.core.TrackingState.TRACKING) {
                    val currentPoseMatrix = FloatArray(16)
                    activeImage.centerPose.toMatrix(currentPoseMatrix, 0)
                    
                    // Temporal smoothing to prevent can-warp drift
                    for (i in 0..15) {
                        smoothedMatrix[i] = smoothedMatrix[i] + SMOOTHING_FACTOR * (currentPoseMatrix[i] - smoothedMatrix[i])
                    }
                    
                    val userOffset = FloatArray(16)
                    Matrix.setIdentityM(userOffset, 0)
                    Matrix.translateM(userOffset, 0, mTranslationX / 100f, mTranslationY / 100f, mTranslationZ / 100f)
                    Matrix.rotateM(userOffset, 0, mRotationX, 1f, 0f, 0f)
                    Matrix.rotateM(userOffset, 0, mRotationY, 0f, 1f, 0f)
                    Matrix.rotateM(userOffset, 0, mRotationZ, 0f, 0f, 1f)
                    Matrix.scaleM(userOffset, 0, mScaleX / 100f, mScaleY / 100f, mScaleZ / 100f)

                    Matrix.multiplyMM(anchorMatrix, 0, smoothedMatrix, 0, userOffset, 0)
                    val viewModelMatrix = FloatArray(16)
                    Matrix.multiplyMM(viewModelMatrix, 0, viewMatrix, 0, anchorMatrix, 0)
                    drawBox(projectionMatrix, viewModelMatrix)
                    
                } else if (isCameraLocked) {
                    val directModelView = FloatArray(16)
                    Matrix.setIdentityM(directModelView, 0)
                    val totalZMeter = -(mManualDepth + mTranslationZ) / 100f
                    Matrix.translateM(directModelView, 0, mTranslationX / 100f, mTranslationY / 100f, totalZMeter)
                    Matrix.rotateM(directModelView, 0, mRotationX, 1f, 0f, 0f)
                    Matrix.rotateM(directModelView, 0, mRotationY, 0f, 1f, 0f)
                    Matrix.rotateM(directModelView, 0, mRotationZ, 0f, 0f, 1f)
                    Matrix.scaleM(directModelView, 0, mScaleX / 100f, mScaleY / 100f, mScaleZ / 100f)

                    // ZERO-DRIFT DRAW
                    drawBox(projectionMatrix, directModelView)

                    // EXPORT ANNOTATION MATRIX
                    val cameraPoseMatrix = FloatArray(16)
                    camera.pose.toMatrix(cameraPoseMatrix, 0)
                    Matrix.multiplyMM(anchorMatrix, 0, cameraPoseMatrix, 0, directModelView, 0)
                } else {
                    // --- WORLD ANCHOR MODE (Standard ARCore Surface Tracking) ---
                    val anchor = currentAnchor
                    if (anchor != null && anchor.trackingState == com.google.ar.core.TrackingState.TRACKING) {
                        // 1. Get the raw anchor pose
                        anchor.pose.toMatrix(anchorMatrix, 0)
                        
                        // 2. Apply transformations DIRECTLY to the anchor matrix (Standard Practice)
                        Matrix.translateM(anchorMatrix, 0, mTranslationX / 100f, mTranslationY / 100f, mTranslationZ / 100f)
                        Matrix.rotateM(anchorMatrix, 0, mRotationX, 1f, 0f, 0f)
                        Matrix.rotateM(anchorMatrix, 0, mRotationY, 0f, 1f, 0f)
                        Matrix.rotateM(anchorMatrix, 0, mRotationZ, 0f, 0f, 1f)
                        Matrix.scaleM(anchorMatrix, 0, mScaleX / 100f, mScaleY / 100f, mScaleZ / 100f)

                        // 3. Draw
                        val viewModelMatrix = FloatArray(16)
                        Matrix.multiplyMM(viewModelMatrix, 0, viewMatrix, 0, anchorMatrix, 0)
                        drawBox(projectionMatrix, viewModelMatrix)
                    }
                }
            }
            
            // Callback to Activity for frame processing if needed
            (context as MainActivity).onFrameRendered(frame, mvpMatrix, anchorMatrix, viewMatrix)

        } catch (t: Throwable) {
            // Avoid crashing
        }
    }

    private fun handleTaps(frame: com.google.ar.core.Frame, camera: com.google.ar.core.Camera) {
        val tap = queuedTaps.poll() ?: return
        if (camera.trackingState != com.google.ar.core.TrackingState.TRACKING) return

        val hits = frame.hitTest(tap)
        // PRIORITY: STRICTLY HORIZONTAL PLANES ONLY for Data Recording Stability
        val bestHit = hits.firstOrNull { hit ->
            val t = hit.trackable
            (t is com.google.ar.core.Plane && t.trackingState == com.google.ar.core.TrackingState.TRACKING && 
             t.type == com.google.ar.core.Plane.Type.HORIZONTAL_UPWARD_FACING)
        }
        // Removed fallback to DepthPoint to prevent drift.

        if (bestHit != null) {
            val hitPose = bestHit.hitPose
            
            // For ground planes, we force UPRIGHT to prevent annoying tilt on placement.
            // For depth points, we use the original hitPose for better handheld alignment.
            val finalPose = if (bestHit.trackable is com.google.ar.core.Plane) {
                 com.google.ar.core.Pose.makeTranslation(hitPose.tx(), hitPose.ty(), hitPose.tz())
            } else {
                hitPose
            }
            
            // GL Thread safe update
            resetAnchor()
            synchronized(this) {
                currentAnchor = bestHit.trackable.createAnchor(finalPose)
            }
            
            (context as MainActivity).runOnUiThread {
                context.onAnchorPlaced()
            }
        }
    }


    private fun drawBox(proj: FloatArray, viewModel: FloatArray) {
        Matrix.multiplyMM(mvpMatrix, 0, proj, 0, viewModel, 0)
        
        GLES30.glUseProgram(program)
        positionHandle = GLES30.glGetAttribLocation(program, "vPosition")
        colorHandle = GLES30.glGetAttribLocation(program, "aColor")
        mvpMatrixHandle = GLES30.glGetUniformLocation(program, "uMVPMatrix")
        tintHandle = GLES30.glGetUniformLocation(program, "uTint")

        // Set Tint
        GLES30.glUniform4fv(tintHandle, 1, mBoxColor, 0)

        if (isCylinderMode) {
            customCylinder?.draw(positionHandle, mvpMatrixHandle, mvpMatrix)
        } else {
            customBox?.draw(positionHandle, colorHandle, mvpMatrixHandle, mvpMatrix)
        }
    }

    private fun loadShader(type: Int, shaderCode: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, shaderCode)
        GLES30.glCompileShader(shader)
        return shader
    }
    
    fun getTextureId(): Int = backgroundRenderer.textureId
    
    fun hasTrackingPlane(): Boolean {
        val session = this.session ?: return false
        return session.getAllTrackables(com.google.ar.core.Plane::class.java).any { 
            it.trackingState == com.google.ar.core.TrackingState.TRACKING &&
            it.type == com.google.ar.core.Plane.Type.HORIZONTAL_UPWARD_FACING
        }
    }
}
