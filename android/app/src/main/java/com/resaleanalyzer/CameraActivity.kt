package com.resaleanalyzer

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.android.material.textfield.TextInputEditText
import com.resaleanalyzer.databinding.ActivityCameraBinding
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var capturedFile: File? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }

        binding.btnCapture.setOnClickListener { takePhoto() }
        binding.btnRetake.setOnClickListener { resetToCapture() }
        binding.btnNext.setOnClickListener { showCostDialog() }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
            } catch (exc: Exception) {
                Toast.makeText(this, "Camera failed: ${exc.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val photoFile = File(cacheDir, "ITEM_$timestamp.jpg")

        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        binding.btnCapture.isEnabled = false

        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    capturedFile = photoFile
                    showPreview(photoFile)
                }

                override fun onError(exc: ImageCaptureException) {
                    binding.btnCapture.isEnabled = true
                    Toast.makeText(
                        this@CameraActivity,
                        "Capture failed: ${exc.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        )
    }

    private fun showPreview(file: File) {
        binding.viewFinder.visibility = View.GONE
        binding.imgPreview.visibility = View.VISIBLE
        binding.btnCapture.visibility = View.GONE
        binding.layoutPostCapture.visibility = View.VISIBLE

        binding.imgPreview.loadFile(file)
    }

    private fun resetToCapture() {
        capturedFile?.delete()
        capturedFile = null
        binding.viewFinder.visibility = View.VISIBLE
        binding.imgPreview.visibility = View.GONE
        binding.btnCapture.visibility = View.VISIBLE
        binding.btnCapture.isEnabled = true
        binding.layoutPostCapture.visibility = View.GONE
    }

    private fun showCostDialog() {
        val file = capturedFile ?: return

        val dialogView = layoutInflater.inflate(R.layout.dialog_cost_input, null)
        val etCost = dialogView.findViewById<TextInputEditText>(R.id.etCost)

        AlertDialog.Builder(this)
            .setTitle("What did you pay?")
            .setMessage("Enter the item's purchase cost")
            .setView(dialogView)
            .setPositiveButton("Analyze") { _, _ ->
                val costText = etCost.text?.toString()?.trim() ?: ""
                val cost = costText.toDoubleOrNull()
                if (cost == null || cost <= 0) {
                    Toast.makeText(this, "Please enter a valid cost", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                launchAnalysis(file, cost)
            }
            .setNegativeButton("Cancel", null)
            .show()

        // Focus and show keyboard
        etCost.requestFocus()
    }

    private fun launchAnalysis(imageFile: File, cost: Double) {
        val intent = Intent(this, AnalysisActivity::class.java).apply {
            putExtra(AnalysisActivity.EXTRA_IMAGE_PATH, imageFile.absolutePath)
            putExtra(AnalysisActivity.EXTRA_COST, cost)
        }
        startActivity(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

private fun android.widget.ImageView.loadFile(file: File) {
    val context = this.context
    val request = coil.request.ImageRequest.Builder(context)
        .data(file)
        .crossfade(true)
        .target(this)
        .build()
    coil.ImageLoader(context).enqueue(request)
}
