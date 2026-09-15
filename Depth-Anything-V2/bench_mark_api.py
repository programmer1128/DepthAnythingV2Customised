from gradio_client import Client, handle_file
import time
import shutil

client = Client("depth-anything/Depth-Anything-V2")

start_time = time.time()

# Pass argument positionally without specifying api_name
result = client.predict(
    handle_file("/home/aritra/Desktop/test.jpg")
)

end_time = time.time()
print(f">> Total cloud pipeline latency: {end_time - start_time:.2f} seconds")

output_path = "api_depth_result.jpg"
shutil.copy(result[0] if isinstance(result, tuple) else result, output_path)
print(f">> Saved to {output_path}")