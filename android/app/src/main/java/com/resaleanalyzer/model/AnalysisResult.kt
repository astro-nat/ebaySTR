package com.resaleanalyzer.model

import com.google.gson.annotations.SerializedName

data class AnalysisResult(
    @SerializedName("item_name") val itemName: String,
    @SerializedName("verdict") val verdict: String,           // "BUY" or "PASS"
    @SerializedName("reason") val reason: String,
    @SerializedName("str_pct") val strPct: Double?,
    @SerializedName("str_source") val strSource: String?,
    @SerializedName("avg_sale_price") val avgSalePrice: Double?,
    @SerializedName("price_low") val priceLow: Double?,
    @SerializedName("price_high") val priceHigh: Double?,
    @SerializedName("comp_count") val compCount: Int,
    @SerializedName("net_proceeds") val netProceeds: Double?,
    @SerializedName("roi_pct") val roiPct: Double?,
    @SerializedName("ebay_fee") val ebayFee: Double?,
    @SerializedName("your_cost") val yourCost: Double,
    @SerializedName("passes_str") val passesStr: Boolean,
    @SerializedName("passes_roi") val passesRoi: Boolean,
)

data class ErrorResponse(
    @SerializedName("error") val error: String
)
