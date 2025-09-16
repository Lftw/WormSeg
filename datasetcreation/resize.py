from PIL import Image
import os

input_dir = "output_folder52_3146_input"
mask_dir = "output_folder52_3146f"
resized_inputs = "output_folder52_resized_inputs"
resized_masks = "output_folder52_resized_masks"
target_size = (224, 224)

os.makedirs(resized_inputs, exist_ok=True)
os.makedirs(resized_masks, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".png") and fname.startswith("frame_"):
        # ----- image -----
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)
        img.save(os.path.join(resized_inputs, fname))

        # ----- mask -----
        # turn "frame_0000.png" -> "0_mask.png"
        num = int(fname.replace("frame_", "").replace(".png", ""))
        mask_name = f"{num}_mask.png"
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(target_size, Image.NEAREST)
            mask.save(os.path.join(resized_masks, mask_name))
        else:
            print(f"Mask not found for {fname} -> expected {mask_name}")
