import os
import time
import pandas as pd
from ultralytics import YOLO

# Configuration
MODELS = {
    "Nano_FP16":   r"E:\AI ML\DRISHTI\models\drishti_v5_nano_web\weights\best_saved_model\best_float16.tflite",
    "Small_FP16":  r"E:\AI ML\DRISHTI\models\drishti_v5_small_app\weights\best_saved_model\best_float16.tflite",
    "Medium_INT8": r"E:\AI ML\DRISHTI\models\Medium_1024_Elite_INT8.tflite"
}

# Just point to the folder with your 'Elite' validation images
IMG_DIR = r"E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\valid\images"
NUM_SAMPLES = 20  # Statistically enough for a hardware speed benchmark

def run_optimized_benchmark():
    results = []
    
    # Get a fixed list of images for all models to ensure fairness
    sample_images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:NUM_SAMPLES]

    print(f"🚀 Benchmarking {len(MODELS)} models on {NUM_SAMPLES} samples each...")

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"⚠️ Missing: {name}")
            continue

        res = 1024 if "Medium" in name else 640
        print(f"🧐 Testing {name} ({res}px)...")

        try:
            model = YOLO(path, task='detect')
            
            # Warm-up (Ensures memory is allocated before timing)
            model.predict(sample_images[0], imgsz=res, verbose=False)

            latencies = []
            for img in sample_images:
                start = time.perf_counter()
                model.predict(img, imgsz=res, verbose=False, conf=0.25)
                latencies.append((time.perf_counter() - start) * 1000)

            avg_inf = sum(latencies) / len(latencies)
            file_size = os.path.getsize(path) / (1024 * 1024)

            results.append({
                "Model": name,
                "Resolution": f"{res}px",
                "Avg Inf (ms)": round(avg_inf, 2),
                "Est. FPS": round(1000 / avg_inf, 1),
                "Size (MB)": round(file_size, 2)
            })
            
            # Help the RAM: Explicitly delete the model object after each run
            del model

        except Exception as e:
            print(f"❌ Error with {name}: {e}")

    # Display the final comparison
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("                DRISHTI PERFORMANCE REPORT")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        df.to_csv("drishti_benchmark_final.csv", index=False)
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_optimized_benchmark()