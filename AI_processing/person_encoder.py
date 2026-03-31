from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt') # Standard model
img_path = '../datasets/weapon-detection-1/valid/images'

# Detect only class 0 (Person) and save as labels
# Added stream=True and removed result accumulation
results = model.predict(source=img_path, save_txt=True, classes=[0], conf=0.5, stream=True)

for result in results:
    # The 'result' object is processed and then eligible for garbage collection
    pass
# Note: YOLO saves these to 'runs/detect/predict/labels'
# You will need to move them back to your dataset/train/labels folder