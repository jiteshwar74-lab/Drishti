import os
import gc
import shutil
import torch
import torch.quantization as tq
from ultralytics import YOLO

os.environ['YOLO_OFFLINE']        = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

QAT_MODEL  = r'E:/AI ML/DRISHTI/models/Medium_1024_Elite_QAT/weights/best.pt'
OUTPUT_DIR = r'E:/AI ML/DRISHTI/models'
MODEL_NAME = 'Medium_1024_Elite_INT8'

def convert_and_export():
    print("🚀 Starting QAT Convert + TFLite Export...")

    # ── 1. Load QAT model ─────────────────────────────────────────────────
    model = YOLO(QAT_MODEL)
    torch_model = model.model

    # ── 2. Convert FakeQuantize → real int8 ops ───────────────────────────
    # MUST be done before export — otherwise you export a float model
    torch_model.eval()
    torch.quantization.convert(torch_model, inplace=True)
    print("✅ FakeQuantize → real int8 ops converted.")

    # ── 3. Clear memory before heavy export ───────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✅ VRAM cleared. Free: " f"{torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    # ── 4. Export to TFLite int8 ──────────────────────────────────────────
    print("\n🛠️  Exporting to TFLite int8 (Flutter target)...")
    try:
        export_path = model.export(
            format='tflite',
            int8=True,
            imgsz=1024,
            data=r'E:/AI ML/DRISHTI/datasets/drishti_full_v4/elite_dataset/elite_drishti.yaml',
            batch=1,
            fraction=0.05,    # uses only 5% of dataset for int8 calibration
            exist_ok=True
        )

        # ── 5. Move to your output folder ─────────────────────────────────
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.tflite")
        shutil.move(str(export_path), final_path)

        print(f"\n✅ SUCCESS!")
        print(f"   📁 Model location : {final_path}")
        print(f"   📱 Next step      : Drop into Flutter assets/ folder")
        print(f"   🔌 Plugin         : tflite_flutter (pub.dev)")

    except MemoryError:
        print("\n❌ RAM OOM during export.")
        print("   Fix: Restart PC to clear RAM bloat, run this script first thing.")

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print("   Fix 1: pip install tensorflow --break-system-packages")
        print("   Fix 2: Restart terminal to clear cached venv state")

if __name__ == '__main__':
    convert_and_export()