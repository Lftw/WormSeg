import h5py
import matplotlib.pyplot as plt

def show_first_frame(file_path, dataset_name):
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name]
        
        # Take the first frame
        frame = dataset[0]
        
        # If it's grayscale (2D), show directly
        if frame.ndim == 2:
            plt.imshow(frame, cmap="gray")
        else:
            plt.imshow(frame)
        
        plt.title(f"First frame of dataset '{dataset_name}'")
        plt.axis("off")
        plt.show()

file_path = "C:/Users/alexw\Documents\SUMMER-25\ZHEN-LAB\Sruthy's recording\output_folder52_without_x1024_1132_without_x0_108.hdf5"

# Example usage
if __name__ == "__main__":
    hdf5_file = file_path
    dataset_name = "images"  # replace with actual dataset name
    show_first_frame(hdf5_file, dataset_name)
