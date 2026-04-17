import os
from collections import Counter

# --- CONFIGURATION ---
labels_path = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\labels'
HUMAN_CLASS_ID = '0'

def audit_labels():
    all_label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    human_only_files = []
    mixed_or_other_files = []
    background_images = 0

    for filename in all_label_files:
        file_path = os.path.join(labels_path, filename)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Filter out empty lines or lines that are just whitespace
            valid_lines = [l.strip() for l in lines if l.strip()]
            
            if not valid_lines:
                background_images += 1
                continue

            # Extract class IDs only from lines that actually have content
            classes_in_file = set(line.split()[0] for line in valid_lines)

            if classes_in_file == {HUMAN_CLASS_ID}:
                human_only_files.append(filename)
            else:
                mixed_or_other_files.append(filename)

    # --- REPORT ---
    print("\n--- Dataset Imbalance Report ---")
    print(f"Total Label Files: {len(all_label_files)}")
    print(f"Images with ONLY Humans: {len(human_only_files)}")
    print(f"Images with Weapons/Mixed: {len(mixed_or_other_files)}")
    print(f"Background (Empty) Images: {background_images}")
    
    with open('human_only_list.txt', 'w') as f:
        for item in human_only_files:
            f.write("%s\n" % item)
            
    print(f"\n[Done] Processed {len(all_label_files)} files successfully.")
    
if __name__ == "__main__":
    audit_labels()