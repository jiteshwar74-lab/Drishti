import os

# CONFIGURATION
IMAGE_DIR = "datasets/drishti_full_v4/images/train"
LABEL_DIR = "datasets/drishti_full_v4/labels/train"

def clean_orphaned_images():
    print("--- Starting Dataset Cleanup ---")
    
    # Get lists of all files
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(LABEL_DIR) if f.endswith('.txt')}
    
    removed_count = 0
    total_images = len(image_files)

    for base_name, full_filename in image_files.items():
        # If the image doesn't have a matching label file
        if base_name not in label_files:
            file_to_remove = os.path.join(IMAGE_DIR, full_filename)
            try:
                os.remove(file_to_remove)
                removed_count += 1
            except Exception as e:
                print(f"Error deleting {full_filename}: {e}")

    print(f"--- Cleanup Complete ---")
    print(f"Total images checked: {total_images}")
    print(f"Orphaned images removed: {removed_count}")
    print(f"Remaining images: {total_images - removed_count}")

if __name__ == "__main__":
    clean_orphaned_images()