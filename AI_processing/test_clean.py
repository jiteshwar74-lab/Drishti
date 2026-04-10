import os
import shutil

# Update these paths to your test set locations
img_dir = r"E:\AI ML\DRISHTI\datasets\weapon-detection-1\test\images"
lbl_dir = r"E:\AI ML\DRISHTI\datasets\weapon-detection-1\test\REMASTERED_LABELS"
junk_dir = r"E:/AI ML/DRISHTI/datasets/test/unlabeled_backup"

os.makedirs(junk_dir, exist_ok=True)

images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
count = 0

for img in images:
    label_name = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(lbl_dir, label_name)
    
    # If the label file doesn't exist OR is empty (size 0)
    if not os.path.exists(label_path) or os.stat(label_path).st_size == 0:
        print(f"Missing label for: {img} -> Moving to backup")
        shutil.move(os.path.join(img_dir, img), os.path.join(junk_dir, img))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(junk_dir, label_name))
        count += 1

print(f"--- Cleanup Complete: {count} images removed ---")