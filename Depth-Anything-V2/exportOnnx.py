import torch
from depth_anything_v2.dpt import DepthAnythingV2

def export_to_onnx():
    # 1. Define the configurations for all encoder sizes
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'
    checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
    onnx_output_path = f'depth_anything_v2_{encoder}.onnx'

    print(f"Loading {encoder} model from {checkpoint_path}...")
    
    # 2. Initialize and load the model on CPU
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # 3. Create a dummy input tensor
    # 518x518 is the optimal resolution for Depth Anything V2.
    # The batch size is 1, and the channels are 3 (RGB).
    dummy_input = torch.randn(1, 3, 518, 518)

    print("Exporting to ONNX... (This might take a couple of minutes for vitl)")

    # 4. Export the graph
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=17,       # Required for advanced ViT operations
        do_constant_folding=True, # Optimizes the graph for inference speed
        input_names=['input'],
        output_names=['depth']
    )
    
    print(f"Success! ONNX model saved to: {onnx_output_path}")

if __name__ == "__main__":
    export_to_onnx()