import os

LABEL_DIR = "E:/AI ML/DRISHTI/datasets/weapon-detection-1/test/REMASTERED_LABELS"

def final_safety_check():
    print("Starting final label cleanup...")
    changed_files = 0
    total_files = 0

    for filename in os.listdir(LABEL_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(LABEL_DIR, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            file_modified = False
            for line in lines:
                parts = line.split()
                if not parts: continue
                
                # If class ID is 3 or higher, force it to 1 (Firearm)
                if int(parts[0]) >= 3:
                    parts[0] = '1'
                    file_modified = True
                
                new_lines.append(" ".join(parts) + "\n")
            
            if file_modified:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                changed_files += 1
            total_files += 1

    print(f"Safety Sweep Complete!")
    print(f"Checked: {total_files} files")
    print(f"Fixed: {changed_files} files containing out-of-bounds indices.")

if __name__ == "__main__":
    final_safety_check()