import os

# --- SETTINGS ---
dual_labels = r"E:\AI ML\DRISHTI\O_datasets\OI_Dual\labels\val"
knives_only_labels = r"E:\AI ML\DRISHTI\O_datasets\OI_Knives_Only\labels\val"

# Map for OI_Dual (where Person=0, Knife=2, Dagger=12, Sword=15)
map_dual = {
    '0': '0',   # Person -> 0
    '2': '2',   # Knife -> 2
    '3': '2',   # Kitchen knife -> 2
    '12': '2',  # Dagger -> 2
    '15': '2'   # Sword -> 2
}

# Map for OI_Knives_Only (where Knife=0, Dagger=1, Kitchen=2, Axe=11)
map_knives = {
    '0': '2',   # Knife -> 2
    '1': '2',   # Dagger -> 2
    '2': '2',   # Kitchen knife -> 2
    '11': '2'   # Axe -> 2 (Great for variety!)
}

def apply_mapping(path, mapping):
    if not os.path.exists(path):
        print(f"Skipping: {path} (Not found)")
        return
    
    print(f"Processing: {path}")
    count = 0
    for filename in os.listdir(path):
        if filename.endswith(".txt") and filename != "classes.txt":
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    old_cls = parts[0]
                    if old_cls in mapping:
                        parts[0] = mapping[old_cls]
                        new_lines.append(" ".join(parts) + "\n")
            
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            count += 1
    print(f"Done! Cleaned {count} files.")

# --- EXECUTION ---
apply_mapping(dual_labels, map_dual)
apply_mapping(knives_only_labels, map_knives)

print("\nAll folders are now synchronized to 0=Person and 2=Knife!")