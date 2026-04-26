package com.example.android_app

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class OverlayView(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    var detections: List<Detection> = emptyList()

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 50f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (det in detections) {

            // Convert normalized → screen coordinates
            val left = (det.x - det.w / 2f) * width
            val top = (det.y - det.h / 2f) * height
            val right = (det.x + det.w / 2f) * width
            val bottom = (det.y + det.h / 2f) * height

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = when (det.cls) {
                0 -> "Person"
                1 -> "Firearm"
                else -> "Knife"
            }

            canvas.drawText(
                "$label ${"%.2f".format(det.conf)}",
                left,
                top - 10,
                textPaint
            )
        }
    }
}