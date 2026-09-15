import rasterio
import numpy as np
import cv2

# Open the raw file you downloaded (after extracting it from the tar.gz)
with rasterio.open("/home/aritra/Desktop/output_hh.tif") as src:
    # Read the elevation band array
    dem_data = src.read(1)

# Print stats to prove your data isn't empty white space
print(f"Minimum Elevation in file: {np.min(dem_data)} meters")
print(f"Maximum Elevation in file: {np.max(dem_data)} meters")

# Convert the float data into a standard 8-bit grayscale image (0-255)
visual_gray = cv2.normalize(dem_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Save it as a normal PNG you can actually see
cv2.imwrite("leh_terrain_normalized.png", visual_gray)
print("Saved 'leh_terrain_normalized.png'. Open this to see your mountains!")
