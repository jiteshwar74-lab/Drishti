import os
from ultralytics import YOLO

def train_model():
    # Ensure the models folder exists (Looking UP one level from AI_processing)
    model_dir = '../models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load the Nano model (best for browser/mobile speed)
    model = YOLO('yolov8n.pt')

    # Training with native augmentations
    results = model.train(
        data='../datasets/weapon-detection-1/data.yaml', 
        epochs=100,         # With 40k images, 50 epochs is plenty
        imgsz=640,         # Standard resolution
        batch=8,          # Adjust to 8 if you get "Out of Memory"
        device=0,          # RTX 3060
        project='../models',      # Everything saves inside /models
        name='drishti_v1',     # Your weights will be in models/drishti_v1/weights/
        # Native Free Augmentations:
        multi_scale=True,      # Varies image size by +/- 33% during training
        label_smoothing=0.1,   # Helps the AI not be "too sure" about messy labels
        box=7.5,               # Increases weight on the "box" accuracy (good for knives)
        cls=0.5,               # Balance for classification
        mosaic=1.0,        # YOLOv8 has this built-in for free!
        mixup=0.1,         # Blends images to help with occlusion
        degrees=10.0,      # Slight rotation
        flipud=0.0,        # Don't flip upside down
        fliplr=0.5,         # 50% chance to flip left-right
        workers=2          # Number of CPU cores used to load images
    )

    # Export for your DRISHTI web app
    model.export(format='tfjs', project='../models', name='drishti_v1_tfjs')

if __name__ == "__main__":
    train_model()