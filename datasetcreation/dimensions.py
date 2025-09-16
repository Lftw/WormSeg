from PIL import Image

#PNG image dimensions extractor

def get_png_dimensions(file_path):
    with Image.open(file_path) as img:
        if img.format != "PNG":
            raise ValueError("File is not a PNG image.")
        return img.size  # (width, height)

# Example usage
if __name__ == "__main__":
    width, height = get_png_dimensions("output_folder52_3146f/0_mask.png")
    print(f"PNG dimensions: {width} x {height}")

# HDF5 dataset frame dimensions extractor

import h5py

def get_hdf5_frame_dimensions(file_path, dataset_name, frame_index=0):
    """
    Get dimensions of one frame from an HDF5 dataset.
    
    Parameters:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset inside the HDF5 file.
        frame_index (int): Index of the frame to check (default=0).
    
    Returns:
        tuple: (height, width) or full shape if channels are present.
    """
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name]
        
        # Ensure the dataset is at least 3D (frames, height, width, ...)
        if dataset.ndim < 3:
            raise ValueError("Dataset does not contain multiple frames.")
        
        frame = dataset[frame_index]
        return frame.shape

def list_datasets(file_path):
    with h5py.File(file_path, "r") as f:
        def print_name(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_name)

file_path = "C:/Users/alexw\Documents\SUMMER-25\ZHEN-LAB\Sruthy's recording\output_folder52_without_x1024_1132_without_x0_108.hdf5"
# Example usage
# if __name__ == "__main__":
#     hdf5_file = file_path
#     dataset_name = "images"  # <-- replace with actual dataset name
#     frame_dims = get_hdf5_frame_dimensions(hdf5_file, dataset_name, frame_index=0)
#     print(f"Frame dimensions: {frame_dims}")