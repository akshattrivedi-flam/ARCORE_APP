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
    private var program = -1
    private var session: Session? = null
    
    private var mvpMatrix = FloatArray(16)
    private var projectionMatrix = FloatArray(16)
    private var viewMatrix = FloatArray(16)
    private var anchorMatrix = FloatArray(16)

    private var positionHandle = -1
    private var colorHandle = -1
    private var mvpMatrixHandle = -1

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
        varying vec4 vColor;
        void main() {
          gl_FragColor = vColor;
        }
    """.trimIndent()

    // Transform properties (centimeters)
    var mScaleX = 6.5f
    var mScaleY = 12.0f
    var mScaleZ = 6.5f
    var mRotationY = 0.0f
    var mTranslationX = 0.0f
    var mTranslationY = 6.0f
    var mTranslationZ = 0.0f

    var currentAnchor: Anchor? = null

    fun setArSession(session: Session) {
        this.session = session
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.1f, 0.1f, 0.1f, 1.0f)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)

        backgroundRenderer.createOnGlThread()
        customBox = CustomBoundingBox()

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

            if (camera.trackingState == com.google.ar.core.TrackingState.TRACKING) {
                val anchor = currentAnchor
                if (anchor != null && anchor.trackingState == com.google.ar.core.TrackingState.TRACKING) {
                    camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100.0f)
                    camera.getViewMatrix(viewMatrix, 0)
                    
                    anchor.pose.toMatrix(anchorMatrix, 0)
                    
                    // Apply manual transforms (converted to meters)
                    Matrix.translateM(anchorMatrix, 0, mTranslationX / 100f, mTranslationY / 100f, mTranslationZ / 100f)
                    Matrix.rotateM(anchorMatrix, 0, mRotationY, 0f, 1f, 0f)
                    Matrix.scaleM(anchorMatrix, 0, mScaleX / 100f, mScaleY / 100f, mScaleZ / 100f)

                    val tempMatrix = FloatArray(16)
                    Matrix.multiplyMM(tempMatrix, 0, viewMatrix, 0, anchorMatrix, 0)
                    Matrix.multiplyMM(mvpMatrix, 0, projectionMatrix, 0, tempMatrix, 0)

                    drawBox(mvpMatrix)
                }
            }
            
            // Callback to Activity for frame processing if needed
            (context as MainActivity).onFrameRendered(frame, mvpMatrix, anchorMatrix, viewMatrix)

        } catch (t: Throwable) {
            // Avoid crashing
        }
    }

    private fun drawBox(mvp: FloatArray) {
        GLES30.glUseProgram(program)
        positionHandle = GLES30.glGetAttribLocation(program, "vPosition")
        colorHandle = GLES30.glGetAttribLocation(program, "aColor")
        mvpMatrixHandle = GLES30.glGetUniformLocation(program, "uMVPMatrix")

        customBox?.draw(positionHandle, colorHandle, mvpMatrixHandle, mvp)
    }

    private fun loadShader(type: Int, shaderCode: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, shaderCode)
        GLES30.glCompileShader(shader)
        return shader
    }
    
    fun getTextureId(): Int = backgroundRenderer.textureId
}
