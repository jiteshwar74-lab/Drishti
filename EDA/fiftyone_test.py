import os
import fiftyone as fo

dataset_path = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset'

dataset = fo.Dataset.from_dir(dataset_dir=dataset_path,
                            dataset_type=fo.types.YOLOv5Dataset,
                            yaml_path=os.path.join(dataset_path, 'elite_drishti.yaml'))

session = fo.launch_app(dataset)