import fiftyone as fo

dataset = fo.Dataset.from_dir(
    dataset_dir=r"E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset",
    dataset_type=fo.types.YOLOv5Dataset,
    yaml_path=r"E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\elite_drishti.yaml",
    name="Elite-Drishti",
    overwrite=True
)

print(dataset)
print(dataset.first())

session = fo.launch_app(dataset, port=5151)
session.wait()

