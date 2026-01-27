package com.example.arcoreapp

import com.google.gson.annotations.SerializedName

data class AnnotationEntry(
    @SerializedName("frame_id") val frameId: Int,
    @SerializedName("image") val image: String,
    @SerializedName("keypoints_2d") val keypoints2d: List<List<Float>>,
    @SerializedName("keypoints_3d") val keypoints3d: List<List<Float>>, // Left-hand space (+Z forward)
    @SerializedName("visibility") val visibility: List<Float>,
    @SerializedName("camera_intrinsics") val cameraIntrinsics: CameraIntrinsics,
    @SerializedName("view_matrix") val viewMatrix: List<Float>,
    @SerializedName("model_matrix") val modelMatrix: List<Float>,
    @SerializedName("point_cloud") val pointCloud: List<List<Float>>?, // Sparse features
    @SerializedName("timestamp") val timestamp: Long
)

data class CameraIntrinsics(
    @SerializedName("fx") val fx: Float,
    @SerializedName("fy") val fy: Float,
    @SerializedName("cx") val cx: Float,
    @SerializedName("cy") val cy: Float,
    @SerializedName("image_width") val imageWidth: Int,
    @SerializedName("image_height") val imageHeight: Int
)

object AnnotationGenerator {

    private val UNIT_CUBE_POINTS = listOf(
        floatArrayOf(0f, 0f, 0f),       // 0: Center
        floatArrayOf(-0.5f, -0.5f, 0.5f), // 1: Front-Bottom-Left
        floatArrayOf(0.5f, -0.5f, 0.5f),  // 2: Front-Bottom-Right
        floatArrayOf(0.5f, 0.5f, 0.5f),   // 3: Front-Top-Right
        floatArrayOf(-0.5f, 0.5f, 0.5f),  // 4: Front-Top-Left
        floatArrayOf(-0.5f, -0.5f, -0.5f),// 5: Back-Bottom-Left
        floatArrayOf(0.5f, -0.5f, -0.5f), // 6: Back-Bottom-Right
        floatArrayOf(0.5f, 0.5f, -0.5f),  // 7: Back-Top-Right
        floatArrayOf(-0.5f, 0.5f, -0.5f)  // 8: Back-Top-Left
    )

    fun createEntry(
        frameId: Int,
        imageName: String,
        modelMatrix: FloatArray,
        viewMatrix: FloatArray,
        points: List<List<Float>>, // Input raw point cloud (world space)
        fx: Float, fy: Float, cx: Float, cy: Float,
        width: Int, height: Int,
        timestamp: Long
    ): AnnotationEntry {
        val keypoints3d = mutableListOf<List<Float>>()
        val keypoints2d = mutableListOf<List<Float>>()
        val visibility = mutableListOf<Float>()

        for (localPt in UNIT_CUBE_POINTS) {
            // Local -> World
            val worldPt4 = FloatArray(4)
            android.opengl.Matrix.multiplyMV(worldPt4, 0, modelMatrix, 0, floatArrayOf(localPt[0], localPt[1], localPt[2], 1.0f), 0)
            val worldPt = floatArrayOf(worldPt4[0], worldPt4[1], worldPt4[2])

            // Project to Camera Space
            val pointCamera4 = FloatArray(4)
            android.opengl.Matrix.multiplyMV(pointCamera4, 0, viewMatrix, 0, worldPt4, 0)
            
            // OBJECTRON SYNC: Use Left-hand rule (+Z forward). 
            // ARCore is Right-hand (-Z forward), so we negate Z.
            keypoints3d.add(listOf(pointCamera4[0], pointCamera4[1], -pointCamera4[2])) 

            // Project to 2D
            val proj = MathUtils.projectPoint(worldPt, viewMatrix, fx, fy, cx, cy, width, height)
            keypoints2d.add(listOf(proj[0], proj[1], proj[2]))

            // Visibility: Buffer depth > 0 and inside frame
            val isVisible = proj[2] > 0 && proj[0] in 0.0..1.0 && proj[1] in 0.0..1.0
            visibility.add(if (isVisible) 1.0f else 0.0f)
        }

        // Convert Point Cloud to Left-hand camera space
        val pointCloudCam = points.map { pt ->
            val ptCam4 = FloatArray(4)
            android.opengl.Matrix.multiplyMV(ptCam4, 0, viewMatrix, 0, floatArrayOf(pt[0], pt[1], pt[2], 1.0f), 0)
            listOf(ptCam4[0], ptCam4[1], -ptCam4[2]) // Negate Z for Objectron compatibility
        }

        return AnnotationEntry(
            frameId = frameId,
            image = imageName,
            keypoints2d = keypoints2d,
            keypoints3d = keypoints3d,
            visibility = visibility,
            cameraIntrinsics = CameraIntrinsics(fx, fy, cx, cy, width, height),
            viewMatrix = viewMatrix.toList(),
            modelMatrix = modelMatrix.toList(),
            pointCloud = pointCloudCam,
            timestamp = timestamp
        )
    }
}
}
