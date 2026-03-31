import os
import shutil

# Paths to your label folders
label_paths = [
    '../datasets/weapon-detection-1/train/labels',
    '../datasets/weapon-detection-1/valid/labels',
    '../datasets/weapon-detection-1/test/labels'
]

for folder in label_paths:
    if os.path.exists(folder):
        # Delete the folder and all its .txt files
        shutil.rmtree(folder)
        # Recreate the empty folder
        os.makedirs(folder)
        print(f"Cleaned: {folder}")

print("All old labels removed. You now have a clean dataset.")