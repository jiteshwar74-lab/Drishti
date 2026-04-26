package com.example.android_app

import android.content.Context
import android.graphics.*
import android.media.Image
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TfliteEngine(context: Context, modelName: String) {

    private val interpreter: Interpreter
    init {
        val model = loadModelFile(context, modelName)

        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }

        interpreter = Interpreter(model, options)
    }

    private val INPUT_SIZE = 1024
    private val NUM_BOXES = 21504
    private val NUM_ELEMENTS = 7

    val isFloat = interpreter.getInputTensor(0).dataType() != org.tensorflow.lite.DataType.INT8

    val inputBuffer = if (isFloat) {
        ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4) // float = 4 bytes
    } else {
        ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3)
    }.apply {
        order(ByteOrder.nativeOrder())
    }

    // For INT8
    private val outputBufferInt8 = Array(1) {
        Array(NUM_ELEMENTS) { ByteArray(NUM_BOXES) }
    }

    // For FLOAT
    private val outputBufferFloat = Array(1) {
        Array(NUM_ELEMENTS) { FloatArray(NUM_BOXES) }
    }

    private fun loadModelFile(context: Context, name: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(name)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    // SAFE YUV → BITMAP (DEVICE-INDEPENDENT)
    private fun imageToBitmap(image: Image): Bitmap {

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)

        // CORRECT ORDER: V then U (NV21 format)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)

        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    // NMS function
    fun iou(a: Detection, b: Detection): Float {
        val x1 = maxOf(a.x - a.w / 2, b.x - b.w / 2)
        val y1 = maxOf(a.y - a.h / 2, b.y - b.h / 2)
        val x2 = minOf(a.x + a.w / 2, b.x + b.w / 2)
        val y2 = minOf(a.y + a.h / 2, b.y + b.h / 2)

        val interArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val areaA = a.w * a.h
        val areaB = b.w * b.h

        return interArea / (areaA + areaB - interArea + 1e-6f)
    }

    fun detect(image: Image, rotation: Int): List<Detection> {

        val bitmap = imageToBitmap(image)

        val matrix = Matrix()
        matrix.postRotate(rotation.toFloat())

        val rotatedBitmap = Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )

        val resized = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resized)

        val resizeScale = minOf(
            INPUT_SIZE.toFloat() / rotatedBitmap.width,
            INPUT_SIZE.toFloat() / rotatedBitmap.height
        )

        val newW = (rotatedBitmap.width * resizeScale).toInt()
        val newH = (rotatedBitmap.height * resizeScale).toInt()

        val dx = (INPUT_SIZE - newW) / 2f
        val dy = (INPUT_SIZE - newH) / 2f

        val rect = RectF(dx, dy, dx + newW, dy + newH)

        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(rotatedBitmap, null, rect, null)
        inputBuffer.clear()

        val inputType = interpreter.getInputTensor(0).dataType()

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {

                val pixel = resized.getPixel(x, y)

                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                if (inputType == org.tensorflow.lite.DataType.INT8) {
                    // INT8 model
                    inputBuffer.put((r - 128).toByte())
                    inputBuffer.put((g - 128).toByte())
                    inputBuffer.put((b - 128).toByte())
                } else {
                    // FLOAT / FP16 model
                    inputBuffer.putFloat(r / 255f)
                    inputBuffer.putFloat(g / 255f)
                    inputBuffer.putFloat(b / 255f)
                }
            }
        }

        inputBuffer.rewind()

        // DEBUG OUTPUT
        val outputType = interpreter.getOutputTensor(0).dataType()
        if (outputType == org.tensorflow.lite.DataType.INT8) {
            interpreter.run(inputBuffer, outputBufferInt8)
        } else {
            interpreter.run(inputBuffer, outputBufferFloat)
        }

        val preds = Array(NUM_ELEMENTS) { FloatArray(NUM_BOXES) }

        if (outputType == org.tensorflow.lite.DataType.INT8) {

            val scale = interpreter.getOutputTensor(0).quantizationParams().scale
            val zero = interpreter.getOutputTensor(0).quantizationParams().zeroPoint

            for (i in 0 until NUM_ELEMENTS) {
                for (j in 0 until NUM_BOXES) {
                    val q = outputBufferInt8[0][i][j].toInt()
                    preds[i][j] = (q - zero) * scale
                }
            }

        } else {
            // FLOAT / FP16 output
            for (i in 0 until NUM_ELEMENTS) {
                for (j in 0 until NUM_BOXES) {
                    preds[i][j] = outputBufferFloat[0][i][j]
                }
            }
        }

//        Log.d("BEST", "class=$bestClass conf=$bestConf")

        val detections = mutableListOf<Detection>()

        val CONF_THRESHOLD = 0.2f

        for (i in 0 until NUM_BOXES) {

            val x = preds[0][i]
            val y = preds[1][i]
            val w = preds[2][i]
            val h = preds[3][i]

            val c0 = preds[4][i]
            val c1 = preds[5][i]
            val c2 = preds[6][i]

            val maxConf = maxOf(c0, c1, c2)

            val cls = when (maxConf) {
                c0 -> 0
                c1 -> 1
                else -> 2
            }

            if (maxConf > CONF_THRESHOLD) {
                detections.add(Detection(x, y, w, h, maxConf, cls))
            }
        }

        val nmsDetections = mutableListOf<Detection>()
        val sorted = detections.sortedByDescending { it.conf }

        val IOU_THRESHOLD = 0.5f

        for (det in sorted) {
            var keep = true

            for (selected in nmsDetections) {
                if (iou(det, selected) > IOU_THRESHOLD) {
                    keep = false
                    break
                }
            }

            if (keep) nmsDetections.add(det)
        }
        Log.d("NMS", "final detections = ${nmsDetections.size}")
        return nmsDetections
    }
}