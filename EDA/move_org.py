import os
import shutil

# Paths
base_dir = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4'
img_train_path = os.path.join(base_dir, 'images', 'train')
lbl_train_path = os.path.join(base_dir, 'labels', 'train')

# Destination for the "Elite" Dataset
elite_img_path = os.path.join(base_dir, 'elite_dataset', 'images')
elite_lbl_path = os.path.join(base_dir, 'elite_dataset', 'labels')

os.makedirs(elite_img_path, exist_ok=True)
os.makedirs(elite_lbl_path, exist_ok=True)

files = os.listdir(img_train_path)
originals_seen = set()
moved_count = 0

print("Starting the Elite Selection...")

for f in files:
    if f.endswith(('.jpg', '.png', '.jpeg')):
        # Get the unique base name (before .rf.)
        base_name = f.split('.rf.')[0]
        
        if base_name not in originals_seen:
            # This is the first time we've seen this scene, let's keep it!
            
            # 1. Move Image
            src_img = os.path.join(img_train_path, f)
            dst_img = os.path.join(elite_img_path, f)
            
            # 2. Move corresponding Label (.txt)
            label_file = f.rsplit('.', 1)[0] + '.txt'
            src_lbl = os.path.join(lbl_train_path, label_file)
            dst_lbl = os.path.join(elite_lbl_path, label_file)
            
            if os.path.exists(src_lbl):
                shutil.copy2(src_img, dst_img) # copy2 preserves metadata
                shutil.copy2(src_lbl, dst_lbl)
                originals_seen.add(base_name)
                moved_count += 1

print(f"Success! Created an elite dataset with {moved_count} unique scenes.")
print(f"Location: {os.path.join(base_dir, 'elite_dataset')}")