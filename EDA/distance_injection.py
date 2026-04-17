import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --- CONFIG ---
backgrounds_dir = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\backgrounds'
weapon_crops_dir = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\crops'
output_images = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\injected\images'
output_labels = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\injected\labels'

def inject_weapons(firearms_count, knives_count, min_scale, max_scale, bucket_name):
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    
    bg_files = [f for f in os.listdir(backgrounds_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_crops = os.listdir(weapon_crops_dir)
    
    firearm_crops = [f for f in all_crops if any(x in f.lower() for x in ['firearm', 'rifle', 'gun'])]
    knife_crops = [f for f in all_crops if 'knife' in f.lower()]
    
    task_list = ([1] * firearms_count) + ([2] * knives_count)
    random.shuffle(task_list)

    print(f"--- Generating {len(task_list)} samples for bucket: {bucket_name} ---")

    for i, current_class in enumerate(task_list):
        bg_file = random.choice(bg_files)
        crop_file = random.choice(firearm_crops if current_class == 1 else knife_crops)
        
        bg = cv2.imread(os.path.join(backgrounds_dir, bg_file))
        crop = Image.open(os.path.join(weapon_crops_dir, crop_file)).convert("RGBA")
        h_bg, w_bg = bg.shape[:2]

        # 1. SCALE
        scale_factor = random.uniform(min_scale, max_scale)
        new_w = int(w_bg * scale_factor)
        aspect = crop.size[1] / crop.size[0]
        crop = crop.resize((new_w, int(new_w * aspect)), Image.Resampling.LANCZOS)
        
        # 2. ROTATION
        crop = crop.rotate(random.randint(0, 360), expand=True)
        
        # 3. ANTI-OVERFITTING (Blur and Jitter for Distant Objects)
        if bucket_name in ["med", "long"]:
            # Slight blur to mimic atmospheric haze
            if random.random() > 0.5:
                crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
            # Brightness jitter for sun/shadow
            enhancer = ImageEnhance.Brightness(crop)
            crop = enhancer.enhance(random.uniform(0.8, 1.2))

        actual_w, actual_h = crop.size 

        # 4. POSITIONING
        x_off = random.randint(0, max(0, w_bg - actual_w))
        y_off = random.randint(int(h_bg * 0.4), max(int(h_bg * 0.4), h_bg - actual_h))

        # 5. PASTE
        bg_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        bg_pil.paste(crop, (x_off, y_off), crop)
        final_img = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

        # 6. COORDS
        x_center = (x_off + actual_w / 2) / w_bg
        y_center = (y_off + actual_h / 2) / h_bg
        yolo_w = actual_w / w_bg
        yolo_h = actual_h / h_bg

        # 7. SAVE with Bucket Prefix
        unique_id = random.randint(1000, 9999)
        name = f"drishti_{bucket_name}_{i}_{unique_id}"
        
        cv2.imwrite(os.path.join(output_images, f"{name}.jpg"), final_img)
        with open(os.path.join(output_labels, f"{name}.txt"), 'w') as f:
            f.write(f"{current_class} {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")

if __name__ == "__main__":
    # BUCKET 1: Close Range (approx 5-15m) -> 5% to 10% width
    inject_weapons(firearms_count=1000, knives_count=1000, min_scale=0.06, max_scale=0.10, bucket_name="close")
    
    # BUCKET 2: Medium Range (approx 20-40m) -> 2% to 4% width
    inject_weapons(firearms_count=1000, knives_count=1000, min_scale=0.02, max_scale=0.04, bucket_name="med")
    
    # BUCKET 3: Extreme Distance (approx 40m+) -> 0.8% to 1.5% width
    inject_weapons(firearms_count=1000, knives_count=1000, min_scale=0.008, max_scale=0.015, bucket_name="long")

    print("--- ALL 6,000 SAMPLES GENERATED SUCCESSFULLY ---")