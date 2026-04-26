import cv2
import numpy as np
import tensorflow as tf

# ===== CONFIG =====
MODEL_PATH = r"E:\AI ML\DRISHTI\models\Medium_1024_Elite_QAT\weights\best_saved_model\best_full_integer_quant.tflite"
IMAGE_PATH = r"E:\AI ML\DRISHTI\Screenshot 2026-04-20 041841.png"

INPUT_SIZE = 1024
CLASS_NAMES = ["person", "Firearm", "knife"]

# ===== LOAD MODEL =====
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

in_scale, in_zero = inp['quantization']
out_scale, out_zero = out['quantization']

# ===== LOAD IMAGE =====
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ===== QUANTIZE INPUT =====
img = img.astype(np.float32) / 255.0
img = img / in_scale + in_zero
img = img.astype(np.int8)

img = np.expand_dims(img, axis=0)

# ===== RUN MODEL =====
interpreter.set_tensor(inp['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(out['index'])[0]
output = (output.astype(np.float32) - out_zero) * out_scale

# ===== ANALYZE OUTPUT =====
detected_classes = set()
max_conf_per_class = {i: 0.0 for i in range(len(CLASS_NAMES))}

for i in range(output.shape[1]):
    scores = output[4:, i]
    cls = np.argmax(scores)
    conf = scores[cls]

    # track max confidence per class
    if conf > max_conf_per_class[cls]:
        max_conf_per_class[cls] = conf

    if conf > 0.1:
        detected_classes.add(cls)

# ===== PRINT RESULTS =====
print("\n=== MODEL OUTPUT ===")

print("Detected classes (conf > 0.1):")
for cls in detected_classes:
    print(f"- {CLASS_NAMES[cls]}")

print("\nMax confidence per class:")
for cls, conf in max_conf_per_class.items():
    print(f"{CLASS_NAMES[cls]}: {conf:.4f}")