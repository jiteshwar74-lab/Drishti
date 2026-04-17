import os
import shutil

# --- CONFIG ---
dataset_images = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\images'
dataset_labels = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\labels'
backgrounds_dir = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\backgrounds'

os.makedirs(backgrounds_dir, exist_ok=True)

def collect_backgrounds():
    img_files = os.listdir(dataset_images)
    count = 0

    print("Searching for null images...")
    
    for img_name in img_files:
        # Get the name without extension (e.g., 'image_01')
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(dataset_labels, f"{base_name}.txt")

        # A "Null" image is one where the label file doesn't exist 
        # OR the label file is completely empty
        is_null = False
        if not os.path.exists(label_path):
            is_null = True
        elif os.path.getsize(label_path) == 0:
            is_null = True

        if is_null:
            src = os.path.join(dataset_images, img_name)
            dst = os.path.join(backgrounds_dir, img_name)
            shutil.copy2(src, dst) # Use copy2 to preserve metadata
            count += 1

    print(f"Success! Found and copied {count} null images to {backgrounds_dir}")

if __name__ == "__main__":
    collect_backgrounds()