import onnx
import os

# 1. Load the model and automatically resolve any external .data files
model = onnx.load("static/model/sketch_classifier.onnx", load_external_data=True)

# 2. Save it as a brand new file, explicitly forbidding external data
fixed_path = "static/model/sketch_classifier_single.onnx"
onnx.save_model(
    model, 
    fixed_path, 
    save_as_external_data=False
)

print(f"✅ Model weights successfully inlined into a single file: {fixed_path}")