import os
# --- SAFETY RAILS FOR C: DRIVE ---
# This forces YOLO to download models/temp files to E: instead of C:
# --- HARD REDIRECT FOR ALL TEMP DATA ---
# Create these folders on E: first if they don't exist!
os.environ['TEMP'] = r'E:\AI_Cache\temp'
os.environ['TMP'] = r'E:\AI_Cache\temp'
os.environ['PIP_CACHE_DIR'] = r'E:\AI_Cache\pip_cache'
os.environ['YOLO_CONFIG_DIR'] = r'E:\AI_Cache\yolo_config'

from ultralytics import YOLO
def train_model():
    # Load the Medium model
    model = YOLO(r'E:\AI ML\DRISHTI\models\Medium_1024_Elite\weights\last.pt') 

    # Start Training
    results = model.train(
        data=r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\elite_drishti.yaml',
        epochs=150,
        imgsz=1024,        # Crucial for 40m/20m targets
        batch=4,           # Start with 4; if it crashes with "OOM", change to 2
        save=True,         # Ensures checkpoints are saved
        save_period=10,    # Saves a permanent checkpoint every 10 epochs for safety
        mosaic=1.0,        # Simulates distance by shrinking images
        scale=0.5,         # Pushes objects further away
        overlap_mask=True,
        resume=True,# Helps when a Close person is blocking a Far person
        multi_scale=False,  # Trains the model to be flexible with distances
        device=0,          # Your RTX 3060
        workers=2,          # Adjust based on your CPU
        exist_ok=True,      # Overwrites if you had a failed start
        project='E:/AI ML/DRISHTI/models',
        name='Medium_1024_Elite',
        amp=True,
        label_smoothing=0.1, # Better for blurry/distant objects
        mixup=0.1,
        box=7.5,           # Boosted for tiny box accuracy
        cls=2.5,           # Boosted to distinguish between rifle/knife shapes
        close_mosaic=10    # Disables mosaic for the last 10 epochs for fine-tuning
    )

# This block is the "Guard" that prevents the crash on Windows
if __name__ == '__main__':
    train_model()