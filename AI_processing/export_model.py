import os
import subprocess

# Disable the internet-seeking behavior of Ultralytics
os.environ['YOLO_OFFLINE'] = 'True'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

def force_export():
    model_path = r"E:/AI ML/DRISHTI/models/drishti_v5_nano_web/weights/best.pt"
    output_dir = r"E:/AI ML/DRISHTI/models"
    
    print("--- Attempting Surgical Export ---")
    
    # We use a subprocess call to 'yolo' directly. 
    # This often bypasses 'ImportErrors' inside a running Python script.
    command = [
        "yolo", "export",
        f"model={model_path}",
        "format=tfjs",
        "imgsz=640",
        f"project={output_dir}",
        "name=drishti_v5_nano"
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"\nSUCCESS! Model should be in {output_dir}/drishti_v5_nano")
    except Exception as e:
        print(f"\nExport failed. Error: {e}")
        print("\nIf you see 'onnx.serialization' again, it means the Virtual Environment")
        print("is cached. Close VS Code and restart the terminal.")

if __name__ == "__main__":
    force_export()