import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model():
    input_model = "/home/aritra/Desktop/OptimisingDepthAnythingV2/Depth-Anything-V2/DepthAnythingCPP/depth_anything_v2_vitl.onnx"
    output_model = "depth_anything_v2_vitl_int8.onnx"

    if not os.path.exists(input_model):
        print(f"Error: Could not find {input_model} in the current directory.")
        print("Make sure you exported the 'vits' (Small) model first!")
        return

    print(f"Loading original model: {input_model}")
    print("Quantizing weights to 8-bit integers...")

    # Perform dynamic quantization
    # QuantType.QUInt8 maps the 32-bit floats to 8-bit unsigned integers
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QUInt8
    )

    print(f"Success! Quantized model saved to: {output_model}")
    
    # Calculate and display the size reduction
    orig_size = os.path.getsize(input_model) / (1024 * 1024)
    quant_size = os.path.getsize(output_model) / (1024 * 1024)
    
    print("-" * 40)
    print(f"Original FP32 Size: {orig_size:.2f} MB")
    print(f"New INT8 Size:      {quant_size:.2f} MB")
    print(f"Reduction:          {((orig_size - quant_size) / orig_size) * 100:.1f}%")
    print("-" * 40)

if __name__ == "__main__":
    quantize_model()