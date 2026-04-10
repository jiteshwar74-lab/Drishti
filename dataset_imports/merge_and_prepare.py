import os
import shutil
import zipfile

# --- CONFIGURATION ---
# 1. UPDATED: Pointing to REMASTERED_LABELS as per your request
OLD_PERSON_IMAGES = "datasets/weapon-detection-1/train/images" 
OLD_PERSON_LABELS = "datasets/weapon-detection-1/train/REMASTERED_LABELS"

# 2. Paths for the new Merged Dataset
MERGED_DIR = "datasets/drishti_full_v4"
MERGED_IMAGES = os.path.join(MERGED_DIR, "images/train")
MERGED_LABELS = os.path.join(MERGED_DIR, "labels/train")

os.makedirs(MERGED_IMAGES, exist_ok=True)
os.makedirs(MERGED_LABELS, exist_ok=True)

# 3. DOWNLOAD KAGGLE DATASET
print("--- Downloading Weapon Dataset ---")
os.system("kaggle datasets download -d raghavnanjappan/weapon-dataset-for-yolov5")

# 4. UNZIP
print("--- Unzipping ---")
if os.path.exists("weapon-dataset-for-yolov5.zip"):
    with zipfile.ZipFile("weapon-dataset-for-yolov5.zip", 'r') as zip_ref:
        zip_ref.extractall("temp_weapons")

# 5. FIND AND FIX KAGGLE LABELS
print("--- Locating and Re-indexing Weapon Labels ---")
target_label_dir = None
for root, dirs, files in os.walk("temp_weapons"):
    if "labels" in dirs and "train" in os.listdir(os.path.join(root, "labels")):
        target_label_dir = os.path.join(root, "labels", "train")
        target_image_dir = os.path.join(root, "images", "train")
        break

if not target_label_dir:
    print("Error: Could not find the labels/train folder inside the unzipped data!")
else:
    # Process Kaggle Labels
    for filename in os.listdir(target_label_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(target_label_dir, filename), 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts: continue
                old_id = int(parts[0])
                # MAPPING: Kaggle Knife(0) -> 2, Kaggle Handgun(1) -> 1
                new_id = 2 if old_id == 0 else 1 
                new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
            
            with open(os.path.join(MERGED_LABELS, "weapon_" + filename), 'w') as f:
                f.writelines(new_lines)

    # 6. COPY WEAPON IMAGES
    print("--- Moving Weapon Images ---")
    for img in os.listdir(target_image_dir):
        shutil.copy(os.path.join(target_image_dir, img), os.path.join(MERGED_IMAGES, "weapon_" + img))

# 7. COPY YOUR OLD DATA (Classes 0, 1, 3)
print("--- Merging Old Person/Weapon/Grenade Data ---")
for img in os.listdir(OLD_PERSON_IMAGES):
    shutil.copy(os.path.join(OLD_PERSON_IMAGES, img), os.path.join(MERGED_IMAGES, "old_" + img))

# Ensure we only grab .txt files from REMASTERED_LABELS
for lbl in os.listdir(OLD_PERSON_LABELS):
    if lbl.endswith(".txt"):
        shutil.copy(os.path.join(OLD_PERSON_LABELS, lbl), os.path.join(MERGED_LABELS, "old_" + lbl))

print(f"--- SUCCESS! Full dataset ready at {MERGED_DIR} ---")