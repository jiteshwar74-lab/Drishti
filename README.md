# 🟢 DRISHTI – Real-Time Threat Detection System

> AI-powered Android application for real-time detection of **weapons and humans** using deep learning.

---

## 🚀 Overview

**DRISHTI** is a real-time computer vision-based surveillance system designed to detect:

* 👤 Person
* 🔫 Firearm
* 🔪 Knife

The app uses a **custom-trained YOLO-based TensorFlow Lite model** and runs directly on Android devices for **on-device inference**.

---

## ⚡ Features

* 🎯 Real-time object detection using camera feed
* 🔁 Dual model support:

  * ⚡ INT8 (Fast)
  * 🎯 FP16 (Accurate)
* 🚨 Smart alarm system for weapon detection
* 📊 Live FPS counter
* 🟢 Futuristic HUD-style UI
* 🔘 Detection ON/OFF control
* 🔊 Alarm ON/OFF toggle
* ⏱️ Activity tracking with persistence (2s logic)
* 📦 Fully offline (no internet required)

---

## 🧠 Tech Stack

* **Android (Kotlin)**
* **TensorFlow Lite**
* **CameraX API**
* **Custom YOLOv8 Model (converted to TFLite)**
* **Quantization (INT8 / FP16)**

---

## 🧪 Models

| Model | Type           | Use Case        |
| ----- | -------------- | --------------- |
| INT8  | Quantized      | Fast, real-time |
| FP16  | Half precision | Higher accuracy |

✔ Models are included in the repository inside:

```
app/src/main/assets/
```

---

## 📊 Dataset

This project uses a **custom combined and curated dataset** built by merging and refining multiple sources.

* Data was **cleaned, balanced, and manually curated**
* Includes classes: *Person, Firearm, Knife*
* Designed specifically for **real-world detection scenarios**

📩 Dataset is not publicly included due to size and licensing constraints.
For access, contact:

**[shubhamppandey1084@gmail.com](mailto:shubhamppandey1084@gmail.com)**

---

## ⚙️ How It Works

1. Camera feed is captured using **CameraX**
2. Frames are preprocessed (resize → normalize)
3. TensorFlow Lite model performs inference
4. Outputs are:

   * Decoded
   * Filtered using confidence threshold
   * Processed using NMS (Non-Max Suppression)
5. Results are rendered as bounding boxes on screen

---

## 🔔 Detection Logic

* If **knife or firearm detected**:

  * Alarm triggers 🔊
  * Status changes to **ACTIVE**
* If no detection:

  * Status resets after 2 seconds

---

## 📦 Installation

### 🔹 APK

1. Download APK from releases
2. Install on Android device
3. Grant camera permission

---

### 🔹 Build from Source

```bash
git clone https://github.com/your-username/drishti.git
```

Open in Android Studio → Run

---

## 🔮 Future Improvements

* 🔴 Visual danger alerts (flashing UI)
* 📳 Vibration feedback
* 🎥 Video recording
* ☁️ Cloud logging
* 📊 Analytics dashboard

---

## 👨‍💻 Author

**Shubham Pandey**
AI/ML Developer | Computer Vision Enthusiast

📩 Email: [Click Here](mailto:shubhamppandey1084@gmail.com)
🔗 Linked In: [Click Here](https://www.linkedin.com/in/shubham-pandey-6a65a524a/)

---

## 🧠 One-Line Summary

> *DRISHTI turns your smartphone into a real-time AI-powered surveillance system.*

---
