import os
from tqdm import tqdm

lbl_dir = r"E:\AI ML\DRISHTI\datasets\drishti_full_v4\labels\train"

# Index: 0=Person, 1=Firearm, 2=Knife
counts = {0: 0, 1: 0, 2: 0}

print("--- Analyzing 42,000 Labels ---")
for filename in tqdm(os.listdir(lbl_dir)):
    if filename.endswith(".txt"):
        with open(os.path.join(lbl_dir, filename), 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    cls = int(parts[0])
                    if cls in counts:
                        counts[cls] += 1

print("\n--- Final Statistics for DRISHTI v5 ---")
print(f"Total People:   {counts[0]}")
print(f"Total Firearms: {counts[1]}")
print(f"Total Knives:   {counts[2]}")