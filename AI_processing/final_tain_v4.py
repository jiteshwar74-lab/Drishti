import os
from ultralytics import YOLO

def train_model():
    # Paths for your setup
    model_dir = '../models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. Load the '16-hour' weights but NOT as a resume
    model = YOLO('E:/AI ML/DRISHTI/runs/models/drishti_v13/weights/last.pt')

    # 2. Training with your custom augmentations + Phase II settings
    results = model.train(
        data='E:/AI ML/DRISHTI/datasets/drishti_full_v4/drishti_v4.yaml', # Point to NEW merged yaml
        epochs=50,             # 50 is the "sweet spot" for fine-tuning 31k images
        imgsz=640,
        batch=8,               # Your RTX 3060 should handle 8 comfortably
        device=0,
        project='../models',
        name='drishti_v4_final',
        
        # --- PHASE II OPTIMIZATIONS ---
        resume=False,          # CRITICAL: Start fresh with new classes
        freeze=10,             # CRITICAL: Protects your 16-hour progress
        cache=False,           # Keep False; 31k images will eat your RAM
        amp=True,              # Speed boost for RTX 3060
        
        # --- YOUR PRO AUGMENTATIONS (Kept from your original) ---
        label_smoothing=0.1,
        box=7.5,
        cls=0.5,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        fliplr=0.5,
        workers=4              # Increased to 4 since your laptop likely has 6-8 cores
    )

    # 3. Export for Web
    model.export(format='tfjs')

if __name__ == "__main__":
    train_model()