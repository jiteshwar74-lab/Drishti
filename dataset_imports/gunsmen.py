import os
import shutil

# --- DIRECTORY PATHS ---
# Update this to where the Kaggle files are currently sitting
SOURCE_DIR = "E:/AI ML/DRISHTI/temp_data/temp_ugorjiir_data/Gunmen Dataset/All" 
FINAL_IMG_DIR = "E:/AI ML/DRISHTI/datasets/drishti_full_v4/images/train"
FINAL_LBL_DIR = "E:/AI ML/DRISHTI/datasets/drishti_full_v4/labels/train"

CLASS_MAP = {'15': '0', '16': '1'}

def forced_sync():
    print(f"Checking source: {SOURCE_DIR}")
    if not os.path.exists(SOURCE_DIR):
        print("ERROR: Source directory not found!")
        return

    files = os.listdir(SOURCE_DIR)
    print(f"Found {len(files)} total files in source.")

    moved_count = 0
    for filename in files:
        if filename.lower().endswith(".txt") and filename != "classes.txt":
            label_path = os.path.join(SOURCE_DIR, filename)
            
            # Try to find the image with ANY casing
            base = os.path.splitext(filename)[0]
            img_file = None
            for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                if os.path.exists(os.path.join(SOURCE_DIR, base + ext)):
                    img_file = base + ext
                    break
            
            if not img_file:
                continue

            # Check and Align Label
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.split()
                if parts and parts[0] in CLASS_MAP:
                    parts[0] = CLASS_MAP[parts[0]]
                    new_lines.append(" ".join(parts) + "\n")
            
            # Only move if it contains our classes
            if new_lines:
                unique_name = "ug_" + base
                # Move Label
                with open(os.path.join(FINAL_LBL_DIR, unique_name + ".txt"), 'w') as f:
                    f.writelines(new_lines)
                # Move Image
                shutil.copy(os.path.join(SOURCE_DIR, img_file), 
                            os.path.join(FINAL_IMG_DIR, unique_name + os.path.splitext(img_file)[1]))
                moved_count += 1

    print(f"Successfully moved {moved_count} new images.")

if __name__ == "__main__":
    forced_sync()