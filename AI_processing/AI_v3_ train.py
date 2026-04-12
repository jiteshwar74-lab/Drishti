import os
from ultralytics import YOLO

def train_dual_models():
    # Use absolute paths where possible to avoid 'relative path' confusion
    dataset_yaml = 'E:/AI ML/DRISHTI/datasets/drishti_full_v4/drishti_v4.yaml'
    project_dir = 'E:/AI ML/DRISHTI/models' # Changed to absolute path

    # --- STEP 1: NANO REFINEMENT (The Web Model) ---
    print("Starting Nano Refinement...")
    # Loading your previous best weights
    nano_model = YOLO('E:/AI ML/DRISHTI/models/drishti_v5_nano_web/weights/last.pt')
    
    nano_model.train(
        data=dataset_yaml,
        epochs=25,         
        imgsz=640,
        batch=4,          
        device=0,
        workers=2,
        freeze=5,         
        project=project_dir,
        name='drishti_v5_nano_web',
        resume=True,
        amp=True,
        exist_ok=True      # Overwrites if folder exists, prevents path errors
    )
    
    # --- STEP 2: SMALL UPGRADE (The App Model) ---
    print("Starting Small Upgrade...")
    
    # FIX: Explicitly point to the weights Nano just created
    # YOLOv8 saves in [project]/[name]/weights/best.pt
    refined_nano_path = os.path.join(project_dir, 'drishti_v5_nano_web', 'weights', 'best.pt')
    
    small_model = YOLO('yolov8s.pt')
    
    small_model.train(
        data=dataset_yaml,
        pretrained=refined_nano_path, # Using the refined Nano weights
        epochs=30,         
        imgsz=640,
        batch=8,           
        device=0,
        workers=2,
        project=project_dir,
        name='drishti_v5_small_app',
        freeze=5,          
        amp=True,
        exist_ok=True
    )

if __name__ == "__main__":
    train_dual_models()