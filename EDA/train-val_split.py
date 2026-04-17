import os
import random
import shutil

# Paths
root_dir = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset'
train_img_dir = os.path.join(root_dir, 'images')
train_lbl_dir = os.path.join(root_dir, 'labels')

val_img_dir = os.path.join(root_dir, 'valid', 'images')
val_lbl_dir = os.path.join(root_dir, 'valid', 'labels')

# Create directories if they don't exist
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Get all images
images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
# Use 10% for validation
num_val = int(len(images) * 0.10)
val_images = random.sample(images, num_val)

print(f"Moving {num_val} images to validation set...")

for img_name in val_images:
    # Move Image
    shutil.move(os.path.join(train_img_dir, img_name), os.path.join(val_img_dir, img_name))
    
    # Move matching Label
    lbl_name = os.path.splitext(img_name)[0] + '.txt'
    if os.path.exists(os.path.join(train_lbl_dir, lbl_name)):
        shutil.move(os.path.join(train_lbl_dir, lbl_name), os.path.join(val_lbl_dir, lbl_name))

print("Split Complete!")