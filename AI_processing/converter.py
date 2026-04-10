import tensorflowjs as tfjs

# Paths
input_path = r"E:/AI ML/DRISHTI/runs/models/drishti_v4_final/weights/best_saved_model"
output_path = r"E:/AI ML/DRISHTI/models/drishti_v2_tfjs"

print("--- Starting Minimalist Conversion ---")

try:
    # Most newer versions only need the input and output paths
    # It will automatically find the 'serving_default' signature
    tfjs.converters.convert_tf_saved_model(input_path, output_path)
    
    print(f"SUCCESS! Model files generated in: {output_path}")
except Exception as e:
    print(f"Failed again. Error details: {e}")