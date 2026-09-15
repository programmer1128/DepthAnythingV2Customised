import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import onnx

# 1. Define the Hugging Face Model Repository
# Replace this with the specific SatDepth/Aerial fine-tune you select
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf" 

print(f"Downloading and loading {MODEL_ID}...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)

# Set the model to evaluation mode (disables dropout layers, locks batch norm)
model.eval()

# 2. Create a dummy tensor
# The model needs a mathematical input to "trace" the execution path.
# We use standard 1 Batch, 3 Channels (RGB), 518x518 Width/Height
batch_size = 1
channels = 3
height = 518
width = 518
dummy_input = torch.randn(batch_size, channels, height, width)

# 3. Export to ONNX
onnx_filename = "sat_depth_optimized.onnx"
print(f"Tracing computational graph and exporting to {onnx_filename}...")

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_filename, 
    export_params=True,        # Store the trained weights inside the model file
    opset_version=14,          # Opset 14 supports modern Vision Transformer math
    do_constant_folding=True,  # Crucial for C++ speed (explained below)
    input_names=['pixel_values'], 
    output_names=['predicted_depth'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'}, 
        'predicted_depth': {0: 'batch_size'}
    }
)

# 4. Validate the exported mathematical graph
print("Validating ONNX graph integrity...")
onnx_model = onnx.load(onnx_filename)
onnx.checker.check_model(onnx_model)
print("Conversion successful! Model is ready for C++ Inference.")