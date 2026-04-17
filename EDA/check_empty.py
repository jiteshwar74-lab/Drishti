import os
from collections import Counter

# Update this path to your "Elite" dataset labels folder
label_path = r'E:\AI ML\DRISHTI\labels'
def analyze_labels(path):
    stats = Counter()
    empty_files = 0
    total_files = 0

    if not os.path.exists(path):
        print(f"Error: Path not found: {path}")
        return

    print(f"Scanning labels in: {path}...\n")

    for file in os.listdir(path):
        if file.endswith('.txt'):
            total_files += 1
            file_path = os.path.join(path, file)
            
            # Check if file is physically empty (0 bytes)
            if os.path.getsize(file_path) == 0:
                empty_files += 1
                continue

            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    empty_files += 1
                    continue
                
                for line in lines:
                    parts = line.split()
                    if parts:
                        class_id = parts[0]
                        stats[class_id] += 1

    print("--- Label Discovery Report ---")
    print(f"Total .txt files found: {total_files}")
    print(f"Empty (Background) files: {empty_files}")
    print(f"Files containing detections: {total_files - empty_files}")
    print("-" * 30)
    print("Class ID Counts:")
    # Sorting by ID for readability
    for cid in sorted(stats.keys()):
        name = "Person" if cid == '0' else "Firearm" if cid == '1' else "Knife" if cid == '2' else "Unknown"
        print(f"  ID {cid} ({name}): {stats[cid]} instances")

if __name__ == "__main__":
    analyze_labels(label_path)