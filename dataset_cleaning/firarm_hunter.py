import os
from ultralytics import YOLO
from tqdm import tqdm

# 1. Load your best model
model = YOLO(r"E:/AI ML/DRISHTI/runs/models/drishti_v4_final/weights/best.pt").to('cuda')

# 2. Paths
img_dir = r"E:\AI ML\DRISHTI\datasets\drishti_full_v4\images\train"
lbl_dir = r"E:\AI ML\DRISHTI\datasets\drishti_full_v4\labels\train"

print("--- Starting Targeted Firearm Recovery ---")

# 3. Process Images
files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for img_name in tqdm(files):
    base_name = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(lbl_dir, base_name + ".txt")
    img_path = os.path.join(img_dir, img_name)
    
    # Check if a label file exists and if it already contains a firearm (index 1)
    has_firearm = False
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            if '1 ' in f.read():
                has_firearm = True
    
    # Only run the model if the image is missing a firearm label
    if not has_firearm:
        results = model(img_path, conf=0.6, verbose=False) # High confidence to keep it 'pure'
        
        new_labels = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                # We ONLY care if the model finds a firearm (index 1)
                if cls == 1:
                    coords = " ".join(map(str, box.xywhn[0].tolist()))
                    new_labels.append(f"1 {coords}")
        
        # If new firearms were found, append them to the file
        if new_labels:
            mode = 'a' if os.path.exists(lbl_path) else 'w'
            with open(lbl_path, mode) as f:
                # Add a newline if appending to an existing file
                if mode == 'a' and os.path.getsize(lbl_path) > 0:
                    f.write("\n")
                f.write("\n".join(new_labels))

print("--- Firearm Recovery Complete ---")