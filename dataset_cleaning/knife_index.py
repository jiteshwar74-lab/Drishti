import os
from tqdm import tqdm

# Path to the NEW 4000 labels ONLY
new_labels_path = r"E:/AI ML/DRISHTI/train/labels"

print(f"--- Re-mapping indices in: {new_labels_path} ---")

files = [f for f in os.listdir(new_labels_path) if f.endswith('.txt')]

for filename in tqdm(files):
    file_path = os.path.join(new_labels_path, filename)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) > 0:
            # Logic: If index is 1 (Roboflow default), change to 2 (Drishti Knife)
            if parts[0] == '0':
                parts[0] = '2'
            new_lines.append(" ".join(parts) + "\n")
            
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

print("--- Success: All index 1s are now index 2s ---")