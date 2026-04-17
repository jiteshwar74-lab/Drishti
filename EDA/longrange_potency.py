import os

# Configuration
label_path = r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\labels'
img_size = 640  # Base for normalization

stats = {'person': {'small': 0, 'medium': 0, 'large': 0},
         'firearm': {'small': 0, 'medium': 0, 'large': 0},
         'knife': {'small': 0, 'medium': 0, 'large': 0}}

def check_potency():
    total_objects = 0
    
    for file in os.listdir(label_path):
        if not file.endswith('.txt'): continue
        
        file_path = os.path.join(label_path, file)
        if os.path.getsize(file_path) == 0: continue # Skip null images

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue
                
                # FIX: We only take the first 5 values, ignoring any extra data
                try:
                    cls = int(float(parts[0]))
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue

                # Calculate pixel area
                pixel_w = w * img_size
                pixel_h = h * img_size
                area = pixel_w * pixel_h
                
                # Mapping Class IDs
                if cls == 0: name = 'person'
                elif cls == 1: name = 'firearm'
                elif cls == 2: name = 'knife'
                else: continue
                
                total_objects += 1
                # COCO Metrics for Scale
                if area < 32**2:
                    stats[name]['small'] += 1
                elif area < 96**2:
                    stats[name]['medium'] += 1
                else:
                    stats[name]['large'] += 1

    print("\n--- Spatial Potency Report (Long Range Readiness) ---")
    print(f"Total Detections Analyzed: {total_objects}")
    for cls_name, counts in stats.items():
        total = sum(counts.values())
        if total == 0: continue
        s_per = (counts['small'] / total) * 100
        m_per = (counts['medium'] / total) * 100
        l_per = (counts['large'] / total) * 100
        print(f"\nClass: {cls_name.upper()}")
        print(f"  SMALL (Distant/Far):  {counts['small']} ({s_per:.1f}%)")
        print(f"  MEDIUM (Mid-Range):   {counts['medium']} ({m_per:.1f}%)")
        print(f"  LARGE (Close-Up):     {counts['large']} ({l_per:.1f}%)")

if __name__ == "__main__":
    check_potency()