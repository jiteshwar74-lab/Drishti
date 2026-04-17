import os

image_path = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\images\train'
files = os.listdir(image_path)

# Roboflow usually formats as: originalname_jpg.rf.hash.jpg
# We want to group by 'originalname'
originals = {}

for f in files:
    base_name = f.split('.rf.')[0] # Gets the part before the hash
    if base_name not in originals:
        originals[base_name] = []
    originals[base_name].append(f)

print(f"Total files: {len(files)}")
print(f"Unique original images: {len(originals)}")
print(f"Average copies per image: {len(files)/len(originals):.2f}")