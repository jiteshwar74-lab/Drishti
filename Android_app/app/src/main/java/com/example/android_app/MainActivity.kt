package com.example.android_app

import android.graphics.Color
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.ImageView
import com.example.android_app.OverlayView
import android.widget.Spinner
import android.widget.Switch
import android.widget.Button
import android.widget.ArrayAdapter
import android.widget.AdapterView
import android.view.View
import android.view.ViewGroup
import android.widget.TextView


class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tflite: TfliteEngine
    private lateinit var debugView: ImageView
    private lateinit var overlayView: OverlayView
    private lateinit var fpsText: TextView
    private lateinit var dateText: TextView
    private lateinit var statusText: TextView
    private lateinit var weaponCountText: TextView
    private lateinit var detectButton: Button
    private lateinit var alarmSwitch: Switch

    private var isDetectionOn = true
    private var isAlarmOn = false

    private var lastActiveTime = 0L

    private lateinit var spinner: Spinner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)
        tflite = TfliteEngine(this, "model_int8.tflite")
        spinner = findViewById(R.id.modelSelector)

        val options = listOf("Fast (INT8)", "Accurate (FP16)")

        val adapter = object : ArrayAdapter<String>(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            listOf("Fast (INT8)", "Accurate (FP16)")
        ) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getView(position, convertView, parent) as TextView
                view.setTextColor(Color.GREEN)
                return view
            }

            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getDropDownView(position, convertView, parent) as TextView
                view.setTextColor(Color.GREEN)
                return view
            }
        }

        spinner.adapter = adapter

        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {

            override fun onItemSelected(
                parent: AdapterView<*>,
                view: View?,
                position: Int,
                id: Long
            ) {

                val modelName = if (position == 0)
                    "model_int8.tflite"
                else
                    "model_fp16.tflite"

                tflite = TfliteEngine(this@MainActivity, modelName)
            }

            override fun onNothingSelected(parent: AdapterView<*>) {}
        }

        previewView = findViewById(R.id.previewView)

        cameraExecutor = Executors.newSingleThreadExecutor()
        overlayView = findViewById(R.id.overlayView)

        fpsText = findViewById(R.id.fpsText)
        dateText = findViewById(R.id.dateText)
        statusText = findViewById(R.id.statusText)
        weaponCountText = findViewById(R.id.weaponCountText)
        detectButton = findViewById(R.id.detectButton)
        alarmSwitch = findViewById(R.id.alarmSwitch)

        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        } else {
            startCamera()
        }

        detectButton.setOnClickListener {
            isDetectionOn = !isDetectionOn
            detectButton.text = if (isDetectionOn) "DETECTION ON" else "DETECTION OFF"
        }

        alarmSwitch.setOnCheckedChangeListener { _, isChecked ->
            isAlarmOn = isChecked
        }

        val handler = android.os.Handler()
        val runnable = object : Runnable {
            override fun run() {
                val time = java.text.SimpleDateFormat(
                    "HH:mm:ss",
                    java.util.Locale.getDefault()
                ).format(java.util.Date())

                dateText.text = time
                handler.postDelayed(this, 1000)
            }
        }
        handler.post(runnable)
    }

    private var mediaPlayer: android.media.MediaPlayer? = null

    private fun playAlarm() {
        if (mediaPlayer == null) {
            mediaPlayer = android.media.MediaPlayer.create(this, R.raw.alarm)
            mediaPlayer?.isLooping = true   // 🔥 IMPORTANT
        }

        if (mediaPlayer?.isPlaying == false) {
            mediaPlayer?.start()
        }
    }

    private fun stopAlarm() {
        if (mediaPlayer?.isPlaying == true) {
            mediaPlayer?.pause()
            mediaPlayer?.seekTo(0)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val image = imageProxy.image
                val rotation = imageProxy.imageInfo.rotationDegrees

                if (image != null) {

                    val start = System.currentTimeMillis()

                    val detections = if (isDetectionOn) {
                        tflite.detect(image, rotation)
                    } else {
                        emptyList()
                    }

                    // Weapon Count
                    var weaponCount = 0
                    var isActive = false

                    for (det in detections) {
                        if (det.cls == 1 || det.cls == 2) {
                            weaponCount++
                            isActive = true
                        }
                    }

                    // Current System Time
                    val currentTime = System.currentTimeMillis()

                    if (isActive) {
                        lastActiveTime = currentTime
                    }

                    val showActive = (currentTime - lastActiveTime) < 2000

                    if (isAlarmOn && showActive) {
                        playAlarm()
                    } else {
                        stopAlarm()
                    }

                    val end = System.currentTimeMillis()
                    val fps = 1000f / (end - start + 1)

                    runOnUiThread {
                        weaponCountText.text = "WEAPONS: $weaponCount"

                        statusText.text = if (showActive) "STATUS: ACTIVE" else "STATUS: IDLE"
                        fpsText.text = "FPS: %.2f".format(fps)
                        overlayView.detections = detections
                        overlayView.invalidate()
                    }
                }

                imageProxy.close()
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == 100 &&
            grantResults.isNotEmpty() &&
            grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        }
    }
}