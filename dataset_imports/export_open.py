import os
import fiftyone as fo
import fiftyone.utils.openimages as fouo # The correct module
from fiftyone import ViewField as F

# 1. Paths to your downloaded data
oi_dir = "C:/Users/ADMIN/fiftyone/open-images-v7/train"
metadata_dir = os.path.join(oi_dir, "metadata")

# 2. EMERGENCY FIX: Create a fake attributes.csv if it's missing
attr_path = os.path.join(metadata_dir, "attributes.csv")
if not os.path.exists(attr_path):
    print("Creating empty attributes.csv to satisfy importer...")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(attr_path, 'w') as f:
        f.write("")

print("Directly importing and splitting the dataset...")

# 3. Import using the correct utility class
dataset = fo.Dataset("OI_Temp_Import")
importer = fouo.OpenImagesV7DatasetImporter(
    dataset_dir=oi_dir,
    label_types=["detections"],
    classes=["Person", "Knife"],
)
dataset.add_importer(importer)

# 4. Create the two distinct Views
dual_view = dataset.match(
    F("ground_truth.detections.label").contains(["Person", "Knife"], all=True)
)

knife_only_view = dataset.match(
    F("ground_truth.detections.label").contains("Knife") & 
    ~F("ground_truth.detections.label").contains("Person")
)

print(f"Found {len(dual_view)} images with both Person & Knife")
print(f"Found {len(knife_only_view)} images with Knife only")

# 5. Export separately to your E: drive
dual_view.export(
    export_dir="E:/AI ML/DRISHTI/O_datasets/OI_Dual",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

knife_only_view.limit(2000).export(
    export_dir="E:/AI ML/DRISHTI/O_datasets/OI_Knives_Only",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

print("--- Both exports complete! ---")