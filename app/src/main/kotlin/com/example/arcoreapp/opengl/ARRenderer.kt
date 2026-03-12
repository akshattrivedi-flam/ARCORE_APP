package com.example.arcoreapp.opengl

import android.content.Context
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import com.google.ar.core.Anchor
import com.google.ar.core.Pose
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
    private val lastTrackedAnchorPose = FloatArray(16).apply { Matrix.setIdentityM(this, 0) }
    private var hasLastTrackedAnchorPose = false

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
    @Volatile var isRecordingCapture = false

    // Bounding Box Color [R, G, B, A]
    @Volatile var mBoxColor = floatArrayOf(1.0f, 0.0f, 0.0f, 0.3f) // Default Red

    fun updateBoxColor(color: FloatArray) {
        mBoxColor = color
    }

    @Volatile var currentAnchor: Anchor? = null
        private set

    @Synchronized
    fun resetAnchor() {
        currentAnchor?.detach()
        currentAnchor = null
        hasLastTrackedAnchorPose = false
    }

    @Synchronized
    fun resetPoseSmoothing() {
        // Marker smoothing removed in ground-anchor-only mode.
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

    @Synchronized
    private fun placeAnchorFromHit(hit: com.google.ar.core.HitResult) {
        resetAnchor()
        currentAnchor = hit.createAnchor()
        currentAnchor?.pose?.toMatrix(lastTrackedAnchorPose, 0)
        hasLastTrackedAnchorPose = true
    }

    @Synchronized
    private fun placeAnchorOnPlane(plane: com.google.ar.core.Plane, pose: Pose) {
        resetAnchor()
        currentAnchor = plane.createAnchor(pose)
        currentAnchor?.pose?.toMatrix(lastTrackedAnchorPose, 0)
        hasLastTrackedAnchorPose = true
    }

    private fun projectPointToPlane(pointPose: Pose, planePose: Pose): FloatArray {
        val p = pointPose.translation
        val p0 = planePose.translation
        val n = FloatArray(3)
        planePose.getTransformedAxis(1, 1.0f, n, 0) // plane normal (up-axis)
        val nn = kotlin.math.sqrt((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).toDouble()).toFloat().coerceAtLeast(1e-8f)
        n[0] /= nn
        n[1] /= nn
        n[2] /= nn
        val vx = p[0] - p0[0]
        val vy = p[1] - p0[1]
        val vz = p[2] - p0[2]
        val signedDist = vx * n[0] + vy * n[1] + vz * n[2]
        return floatArrayOf(
            p[0] - signedDist * n[0],
            p[1] - signedDist * n[1],
            p[2] - signedDist * n[2]
        )
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

                if (isCameraLocked) {
                    resetPoseSmoothing()
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
                    resetPoseSmoothing()
                    // --- WORLD ANCHOR MODE (Standard ARCore Surface Tracking) ---
                    val anchor = currentAnchor
                    if (anchor != null) {
                        val baseAnchorPose = FloatArray(16)
                        val hasBasePose = if (anchor.trackingState == com.google.ar.core.TrackingState.TRACKING) {
                            anchor.pose.toMatrix(baseAnchorPose, 0)
                            System.arraycopy(baseAnchorPose, 0, lastTrackedAnchorPose, 0, 16)
                            hasLastTrackedAnchorPose = true
                            true
                        } else if (hasLastTrackedAnchorPose) {
                            System.arraycopy(lastTrackedAnchorPose, 0, baseAnchorPose, 0, 16)
                            true
                        } else {
                            false
                        }

                        if (hasBasePose) {
                            val userOffset = FloatArray(16)
                            Matrix.setIdentityM(userOffset, 0)
                            Matrix.translateM(userOffset, 0, mTranslationX / 100f, mTranslationY / 100f, mTranslationZ / 100f)
                            Matrix.rotateM(userOffset, 0, mRotationX, 1f, 0f, 0f)
                            Matrix.rotateM(userOffset, 0, mRotationY, 0f, 1f, 0f)
                            Matrix.rotateM(userOffset, 0, mRotationZ, 0f, 0f, 1f)
                            Matrix.scaleM(userOffset, 0, mScaleX / 100f, mScaleY / 100f, mScaleZ / 100f)

                            Matrix.multiplyMM(anchorMatrix, 0, baseAnchorPose, 0, userOffset, 0)
                            val viewModelMatrix = FloatArray(16)
                            Matrix.multiplyMM(viewModelMatrix, 0, viewMatrix, 0, anchorMatrix, 0)
                            drawBox(projectionMatrix, viewModelMatrix)
                        }
                    }
                }
            }
            
            // Callback to Activity for frame processing if needed
            (context as MainActivity).onFrameRendered(frame, anchorMatrix, viewMatrix)

        } catch (t: Throwable) {
            // Avoid crashing
        }
    }

    private fun handleTaps(frame: com.google.ar.core.Frame, camera: com.google.ar.core.Camera) {
        val tap = queuedTaps.poll() ?: return
        if (camera.trackingState != com.google.ar.core.TrackingState.TRACKING) return

        val hits = frame.hitTest(tap)
        val horizontalPlaneHit = hits.firstOrNull { hit ->
            val t = hit.trackable
            t is com.google.ar.core.Plane &&
                t.trackingState == com.google.ar.core.TrackingState.TRACKING &&
                t.type == com.google.ar.core.Plane.Type.HORIZONTAL_UPWARD_FACING &&
                t.isPoseInPolygon(hit.hitPose)
        } ?: return

        val plane = horizontalPlaneHit.trackable as com.google.ar.core.Plane
        val depthHit = hits.firstOrNull { hit ->
            val t = hit.trackable
            t is com.google.ar.core.DepthPoint &&
                t.trackingState == com.google.ar.core.TrackingState.TRACKING
        }

        if (depthHit != null) {
            // Ground-anchor strategy:
            // Use depth at the tap ray (can surface) but project onto the horizontal plane
            // so the anchor stays physically stable and upright on the ground/table.
            val projected = projectPointToPlane(depthHit.hitPose, plane.centerPose)
            val planeRotation = horizontalPlaneHit.hitPose.rotationQuaternion
            val anchorPose = Pose.makeTranslation(projected[0], projected[1], projected[2]).compose(
                Pose.makeRotation(
                    planeRotation[0],
                    planeRotation[1],
                    planeRotation[2],
                    planeRotation[3]
                )
            )
            placeAnchorOnPlane(plane, anchorPose)
        } else {
            placeAnchorFromHit(horizontalPlaneHit)
        }

        (context as MainActivity).runOnUiThread {
            context.onAnchorPlaced()
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
