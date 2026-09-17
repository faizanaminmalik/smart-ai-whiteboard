import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define a lightweight CNN architecture for 28x28 sketches
class SketchClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SketchClassifier, self).__init__()
        # Input: 1 channel (grayscale), Output: 16 channels, Kernel: 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Input: 16 channels, Output: 32 channels, Kernel: 3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 32 channels * 7 * 7 spatial dimensions (after pooling twice)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Size drops to 14x14
        x = self.pool(F.relu(self.conv2(x))) # Size drops to 7x7
        x = x.view(-1, 32 * 7 * 7)           # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Instantiate the model
model = SketchClassifier(num_classes=6)

# Note: In a real-world scenario, you would load pre-trained weights here:
# model.load_state_dict(torch.load('my_quickdraw_weights.pth'))
model.eval() # Set model to evaluation mode

# 3. Create a dummy input tensor matching the frontend shape (Batch=1, Channels=1, 28x28)
dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)

# 4. Ensure the output directory exists
os.makedirs("static/model", exist_ok=True)
onnx_path = "static/model/sketch_classifier.onnx"

# 5. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✅ Successfully generated ONNX model at: {onnx_path}")