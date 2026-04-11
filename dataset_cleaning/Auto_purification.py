import os
from ultralytics import YOLO
from tqdm import tqdm

# 1. Load the heavyweight teacher (X-model is best for distinguishing small knives)
model = YOLO('yolov8x.pt').to('cuda')

# 2. Paths
img_dir = r"E:\AI ML\DRISHTI\datasets\drishti_full_v4\images\train"
lbl_dir = r"E:\AI ML\DRISHTI\datasets\drishti_full_v4\labels\train"

print("--- Starting Full-Spectrum Auditor (Fix, Find, & Verify) ---")

files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
stats = {"fixed": 0, "added": 0}

for img_name in tqdm(files):
    base_name = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(lbl_dir, base_name + ".txt")
    img_path = os.path.join(img_dir, img_name)
    
    existing_labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            existing_labels = [line.strip().split() for line in f.readlines()]
            
    existing_indices = {int(x[0]) for x in existing_labels if x}

    # Run the Heavyweight Teacher
    results = model(img_path, conf=0.5, iou=0.45, verbose=False)
    
    teacher_people = [] # List of [x, y, w, h]
    teacher_knives = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            coords = box.xywhn[0].tolist()
            if cls == 0: teacher_people.append(coords)    # Person
            if cls == 43: teacher_knives.append(coords)   # Knife (COCO index)

    new_label_lines = []
    modified = False

    # STEP A: Check/Verify existing labels
    for lbl in existing_labels:
        # Check if the line is empty or just whitespace
        if not lbl or len(lbl) < 5: 
            continue
            
        cls = int(lbl[0])
        our_x, our_y = float(lbl[1]), float(lbl[2])
        
        if cls == 0: # This is a suspected Person label
            # Check if Teacher thinks this specific box is actually a knife
            is_actually_knife = any(abs(our_x - t[0]) < 0.05 and abs(our_y - t[1]) < 0.05 for t in teacher_knives)
            
            if is_actually_knife:
                lbl[0] = '2' # Swap Person to Knife
                modified = True
                stats["fixed"] += 1
        
        new_label_lines.append(" ".join(lbl))

    # STEP B: Add missing Knives (2) or People (0)
    # We only add if the Teacher is very sure and we don't already have a box there
    for t_knife in teacher_knives:
        # If we don't have a knife at these coordinates, add it
        if not any(abs(t_knife[0] - float(l.split()[1])) < 0.05 for l in new_label_lines if l.startswith('2')):
            new_label_lines.append(f"2 {t_knife[0]} {t_knife[1]} {t_knife[2]} {t_knife[3]}")
            modified = True
            stats["added"] += 1

    for t_person in teacher_people:
        # If we don't have a person at these coordinates, add it
        if not any(abs(t_person[0] - float(l.split()[1])) < 0.05 for l in new_label_lines if l.startswith('0')):
            new_label_lines.append(f"0 {t_person[0]} {t_person[1]} {t_person[2]} {t_person[3]}")
            modified = True
            stats["added"] += 1

    # Save the updated file
    if modified:
        with open(lbl_path, 'w') as f:
            f.write("\n".join(new_label_lines))

print(f"--- Audit Finished ---")
print(f"Swapped (0->2): {stats['fixed']} | Newly Labeled: {stats['added']}")