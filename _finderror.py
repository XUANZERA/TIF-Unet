import os
import numpy as np
import tifffile
from PIL import Image

path = "/home/ubuntu/disk1/lzx/dataset/neuron/images_croped512"

error = 0

for filename in os.listdir(path):
    if filename.endswith(".tif"):
        filepath = os.path.join(path, filename)
        img = tifffile.imread(filepath)
        # img = np.array(img)
        print(f"[DEBUG] {filename} shape: {img.shape}, dtype: {img.dtype}")

        if img.ndim != 3 or img.shape != (8, 512, 512):
            print(f"[ERROR] Unexpected shape {img.shape} in file: {filename}")
            error += 1
        
        try:
            pil = Image.fromarray(img)
        except Exception as e:
            print(f"[PIL ERROR] File {filename} failed: {e}")

print(f"Total files with error: {error}")
