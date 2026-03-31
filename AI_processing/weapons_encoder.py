import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# --- CONFIG ---
IMAGE_DIR = '../datasets/weapon-detection-1/train/images'
FINAL_OUTPUT = '../datasets/weapon-detection-1/train/REMASTERED_LABELS'

TEACHER_MAP = {
    0: 1,  # Teacher Gun     -> Your Gun (1)
    2: 3,  # Teacher Grenade -> Your Grenade (3)
    3: 2,  # Teacher Knife   -> Your Knife (2)
}

os.makedirs(FINAL_OUTPUT, exist_ok=True)

# --- LOAD BOTH MODELS ---
weapon_model_path = hf_hub_download(repo_id="Subh775/Threat-Detection-YOLOv8n", filename="weights/best.pt")
weapon_model = YOLO(weapon_model_path)
person_model = YOLO('yolov8n.pt')  # COCO pretrained — Person is class 0

def remaster():
    all_images = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Total images to process: {len(all_images)}")

    stats = {"Person": 0, "Gun": 0, "Knife": 0, "Grenade": 0}
    total = 0

    weapon_stream = weapon_model.predict(source=all_images, conf=0.35, stream=True, device=0)
    person_stream = person_model.predict(source=all_images, conf=0.45, stream=True, device=0)

    for w_result, p_result in zip(weapon_stream, person_stream):
        final_lines = []

        # 1. Person detections from COCO model
        for box in p_result.boxes:
            if int(box.cls[0]) == 0:
                coords = box.xywhn[0].tolist()
                final_lines.append(f"0 {' '.join(map(str, coords))}\n")
                stats["Person"] += 1

        # 2. Weapon detections from teacher model
        for box in w_result.boxes:
            cls = int(box.cls[0])
            if cls in TEACHER_MAP:
                target_cls = TEACHER_MAP[cls]
                coords = box.xywhn[0].tolist()
                final_lines.append(f"{target_cls} {' '.join(map(str, coords))}\n")

                if target_cls == 1: stats["Gun"] += 1
                elif target_cls == 2: stats["Knife"] += 1
                elif target_cls == 3: stats["Grenade"] += 1

        # 3. Always write — empty file = valid background sample for YOLO
        stem = os.path.splitext(os.path.basename(w_result.path))[0]
        with open(os.path.join(FINAL_OUTPUT, stem + '.txt'), 'w') as f:
            f.writelines(final_lines)

        total += 1
        if total % 1000 == 0:
            print(f"Progress: {total}/{len(all_images)} | {stats}")

    print(f"\n✅ REMASTER COMPLETE")
    print(f"Total Images: {total}")
    print(f"\nTotal Instances Found:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    remaster()