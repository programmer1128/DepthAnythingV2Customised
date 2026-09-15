import modal
import subprocess
import socket
import time
from fastapi import FastAPI, Request, Response

web_app = FastAPI()

# Define base path matching your teammate's environment
base_nvidia = "/system/conda/miniconda3/envs/cloudspace/lib/python3.12/site-packages/nvidia"

image = (
    modal.Image.from_registry(
        "ghcr.io/satadru345/depth-pipeline:latest", 
        add_python="3.12"
    )
    .pip_install("fastapi")
    .apt_install("g++", "cmake", "make")
    .run_commands(
        # Create build directory directly under /DepthAnythingCPP
        "cd /DepthAnythingCPP && mkdir -p build && cd build && cmake .. && make -j$(nproc)"
    )
    .env({
        "BASE_NVIDIA": base_nvidia,
        "LD_LIBRARY_PATH": f"{base_nvidia}/cudnn/lib:{base_nvidia}/cuda_runtime/lib:{base_nvidia}/cublas/lib", 
        "OPENCV_IO_MAX_IMAGE_PIXELS": "1099511627776"
    })
)
# 2. Make sure the app is using this specific image!
app = modal.App("depth-pipeline-app", image=image)

@app.function(gpu="any")
@modal.asgi_app()
def web_endpoint():
    subprocess.Popen(["./depth_pipeline"], cwd="/DepthAnythingCPP/build")
    time.sleep(3) 
    return web_app

@web_app.post("/")
async def process_tile(request: Request):
    raw_data = await request.body()
    
    # Open a TCP connection to the internal C++ server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 9092))
    
    # Forward the raw HTTP bytes (Header + Payload) to the C++ socket
    client.sendall(raw_data)
    
    # Read the returning float matrix from the C++ socket
    expected_bytes = 518 * 518 * 4 # 518x518 floats (4 bytes per float)
    result = bytearray()
    
    while len(result) < expected_bytes:
        packet = client.recv(expected_bytes - len(result))
        if not packet: 
            break
        result.extend(packet)
        
    client.close()
    
    # Return the raw matrix bytes via HTTP back to your local proxy
    return Response(content=bytes(result), media_type="application/octet-stream")