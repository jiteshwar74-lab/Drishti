import os

# Path to the Simuletic labels folder
label_path = r"E:\AI ML\DRISHTI\O_datasets\Simuletic_CCTV\Knife_Dataset\labels"

def reindex_simuletic(path):
    print(f"Syncing Simuletic labels to DRISHTI standards...")
    count = 0
    if not os.path.exists(path):
        print(f"Error: Path not found! {path}")
        return

    for filename in os.listdir(path):
        # Skip classes.txt if it exists in the labels folder
        if filename.endswith(".txt") and filename != "classes.txt":
            file_path = os.path.join(path, filename)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    old_id = parts[0]
                    # Map 1 (Knife) -> 2 (DRISHTI Knife)
                    if old_id == '1':
                        parts[0] = '2'
                        new_lines.append(" ".join(parts) + "\n")
                    # Map 0 (Person) -> 0 (DRISHTI Person)
                    elif old_id == '0':
                        new_lines.append(line) 
                    # Optional: keep other classes if you want, or ignore
            
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            count += 1
    print(f"Successfully updated {count} files.")

reindex_simuletic(label_path)