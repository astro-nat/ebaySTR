package com.resaleanalyzer

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.resaleanalyzer.databinding.ActivityAnalysisBinding
import com.resaleanalyzer.model.AnalysisResult
import com.resaleanalyzer.network.ApiService
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.abs

class AnalysisActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_IMAGE_PATH = "extra_image_path"
        const val EXTRA_COST = "extra_cost"
    }

    private lateinit var binding: ActivityAnalysisBinding
    private val apiService = ApiService()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAnalysisBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)
        val cost = intent.getDoubleExtra(EXTRA_COST, 0.0)

        if (imagePath == null || cost <= 0) {
            Toast.makeText(this, "Invalid data — please try again", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val imageFile = File(imagePath)
        loadImageIntoView(imageFile)

        showLoading()
        runAnalysis(imageFile, cost)

        binding.btnScanAnother.setOnClickListener {
            // Pop back to camera
            val intent = Intent(this, CameraActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP
            startActivity(intent)
            finish()
        }
    }

    private fun runAnalysis(imageFile: File, cost: Double) {
        lifecycleScope.launch {
            try {
                val result = apiService.analyze(imageFile, cost)
                showResult(result)
            } catch (e: Exception) {
                showError(e.message ?: "Unknown error")
            }
        }
    }

    private fun showLoading() {
        binding.layoutLoading.visibility = View.VISIBLE
        binding.layoutResult.visibility = View.GONE
        binding.layoutError.visibility = View.GONE
    }

    private fun showResult(result: AnalysisResult) {
        binding.layoutLoading.visibility = View.GONE
        binding.layoutError.visibility = View.GONE
        binding.layoutResult.visibility = View.VISIBLE

        val isBuy = result.verdict == "BUY"

        // Verdict banner
        binding.tvVerdict.text = if (isBuy) "✓ BUY" else "✗ PASS"
        binding.tvVerdict.setBackgroundColor(
            if (isBuy) Color.parseColor("#2E7D32") else Color.parseColor("#C62828")
        )

        // Item name
        binding.tvItemName.text = result.itemName

        // STR row
        val strText = result.strPct?.let { "%.1f%%".format(it) } ?: "N/A"
        val strPass = result.passesStr
        binding.tvStr.text = strText
        binding.tvStr.setTextColor(if (strPass) Color.parseColor("#2E7D32") else Color.parseColor("#C62828"))
        binding.tvStrLabel.text = "Sell-Through Rate ${if (strPass) "✓" else "✗ (need ≥80%)"}"

        // Pricing rows
        val avgPrice = result.avgSalePrice
        binding.tvAvgSalePrice.text = avgPrice?.let { "$${"%.2f".format(it)}" } ?: "N/A"
        binding.tvPriceRange.text = if (result.priceLow != null && result.priceHigh != null) {
            "${"%.0f".format(result.priceLow)} – $${"%.0f".format(result.priceHigh)} (${result.compCount} comps)"
        } else "—"

        binding.tvYourCost.text = "$${"%.2f".format(result.yourCost)}"
        binding.tvEbayFee.text = result.ebayFee?.let { "$${"%.2f".format(it)}" } ?: "—"
        binding.tvNetProceeds.text = result.netProceeds?.let { "$${"%.2f".format(it)}" } ?: "—"

        // ROI row
        val roiText = result.roiPct?.let { "%.0f%%".format(abs(it)) } ?: "N/A"
        val roiPass = result.passesRoi
        binding.tvRoi.text = roiText
        binding.tvRoi.setTextColor(if (roiPass) Color.parseColor("#2E7D32") else Color.parseColor("#C62828"))
        binding.tvRoiLabel.text = "ROI after fees ${if (roiPass) "✓" else "✗ (need ≥500%)"}"

        // Reason / explanation
        binding.tvReason.text = result.reason
        binding.tvReason.visibility = if (!isBuy) View.VISIBLE else View.GONE
    }

    private fun loadImageIntoView(file: File) {
        val request = coil.request.ImageRequest.Builder(this)
            .data(file)
            .crossfade(true)
            .target(binding.imgItem)
            .build()
        coil.ImageLoader(this).enqueue(request)
    }

    private fun showError(message: String) {
        binding.layoutLoading.visibility = View.GONE
        binding.layoutResult.visibility = View.GONE
        binding.layoutError.visibility = View.VISIBLE
        binding.tvError.text = "Analysis failed:\n$message"
        binding.btnRetryAnalysis.setOnClickListener { finish() }
    }
}
