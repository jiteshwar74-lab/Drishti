import os
from collections import Counter

# Using r'' for Windows paths to avoid escape character issues
label_path = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\labels'
stats = Counter()
empty_files = 0
total_processed = 0

print("Scanning files... please wait.")

# scandir is much faster for large directories
with os.scandir(label_path) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.endswith('.txt'):
            total_processed += 1
            
            # Use the entry.stat() which is already cached by scandir
            if entry.stat().st_size == 0:
                empty_files += 1
                continue
            
            with open(entry.path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        stats[parts[0]] += 1
            
            # Print progress every 5000 files so you know it hasn't crashed
            if total_processed % 5000 == 0:
                print(f"Processed {total_processed} files...")

print(f"\n--- Final Report ---")
print(f"Total Files Scanned: {total_processed}")
print(f"Background (Null) Images: {empty_files}")
print(f"Class Distribution: {dict(stats)}")