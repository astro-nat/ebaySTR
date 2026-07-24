package com.resaleanalyzer.network

import android.util.Base64
import com.google.gson.Gson
import com.resaleanalyzer.BuildConfig
import com.resaleanalyzer.model.AnalysisResult
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

class ApiService {

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)   // Vision + eBay lookup can be slow
        .writeTimeout(30, TimeUnit.SECONDS)
        .addInterceptor(
            HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BASIC
            }
        )
        .build()

    private val gson = Gson()
    private val baseUrl = BuildConfig.API_BASE_URL

    /**
     * Send the captured image (as a JPEG file) and purchase cost to the backend.
     * Returns an [AnalysisResult] on success, or throws an exception with a
     * human-readable message on failure.
     */
    suspend fun analyze(imageFile: File, cost: Double): AnalysisResult {
        val imageBytes = imageFile.readBytes()
        val imageB64 = Base64.encodeToString(imageBytes, Base64.NO_WRAP)

        val bodyJson = JSONObject().apply {
            put("image", imageB64)
            put("cost", cost)
        }.toString()

        val request = Request.Builder()
            .url("$baseUrl/analyze")
            .post(bodyJson.toRequestBody("application/json".toMediaType()))
            .build()

        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                val errorMsg = try {
                    JSONObject(body).optString("error", "Server error ${response.code}")
                } catch (_: Exception) {
                    "Server error ${response.code}"
                }
                throw Exception(errorMsg)
            }

            gson.fromJson(body, AnalysisResult::class.java)
                ?: throw Exception("Empty response from server")
        }
    }

    /** Quick health check — returns true if the server is reachable. */
    suspend fun isServerReachable(): Boolean {
        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            try {
                val request = Request.Builder().url("$baseUrl/health").build()
                client.newCall(request).execute().isSuccessful
            } catch (_: Exception) {
                false
            }
        }
    }
}
