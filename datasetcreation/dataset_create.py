import h5py
import numpy as np
import os
from PIL import Image

def save_right_half_frames(hdf5_file, dataset_name, output_folder):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    with h5py.File(hdf5_file, "r") as f:
        dataset = f[dataset_name]
        
        num_frames = dataset.shape[0]
        print(f"Dataset '{dataset_name}' has {num_frames} frames.")
        
        for i in range(num_frames):
            frame = dataset[i]
            
            # Handle grayscale vs RGB
            if frame.ndim == 2:  # grayscale
                h, w = frame.shape
                right_half = frame[:, w//2:]
                img = Image.fromarray(right_half)
            else:  # color (H, W, C)
                h, w, c = frame.shape
                right_half = frame[:, w//2:, :]
                img = Image.fromarray(right_half)
            
            # Save PNG
            out_path = os.path.join(output_folder, f"frame_{i:04d}.png")
            img.save(out_path)
            
            if i % 10 == 0:
                print(f"Saved {i+1}/{num_frames} frames...")
    
    print(f"✅ Done! Saved {num_frames} right-half frames to '{output_folder}'.")

file_path = "C:/Users/alexw\Documents\SUMMER-25\ZHEN-LAB\Sruthy's recording\output_folder52_without_x1024_1132_without_x0_108.hdf5"


# Example usage
if __name__ == "__main__":
    hdf5_file = file_path
    dataset_name = "images"  # replace with actual dataset name
    output_folder = "output_folder52_3146_input"
    
    save_right_half_frames(hdf5_file, dataset_name, output_folder)
