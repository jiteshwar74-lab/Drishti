import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# 1. Download images that contain AT LEAST one of these classes
# We download both to ensure we have the candidate pool
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    classes=["Person", "Knife"], 
    label_types=["detections"],
    max_samples=8000, # Initial pool size
    dataset_dir="E:/AI ML/DRISHTI/OI_Raw"
)

# 2. Filter for images that contain BOTH classes in the same frame
# This is the 'Drishti' secret sauce
dual_view = dataset.match(
    F("ground_truth.detections.label").contains(["Person", "Knife"], all=True)
)

# 3. Filter for images that contain ONLY Knives (to boost your low count)
knife_only_view = dataset.match(
    F("ground_truth.detections.label").contains("Knife") & 
    ~F("ground_truth.detections.label").contains("Person")
)

# 4. Export them into YOLO format
# We'll export the 'Dual' view and the 'Knife Only' view separately or together
print(f"Found {len(dual_view)} images with both Person & Knife")

dual_view.export(
    export_dir="E:/AI ML/DRISHTI/O_datasets/OI_Dual",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

knife_only_view.limit(2000).export( # Just take an extra 2000 pure knives
    export_dir="E:/AI ML/DRISHTI/O_datasets/OI_Knives_Only",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)