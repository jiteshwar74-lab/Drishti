package com.example.android_app

data class Detection(
    val x: Float,
    val y: Float,
    val w: Float,
    val h: Float,
    val conf: Float,
    val cls: Int
)