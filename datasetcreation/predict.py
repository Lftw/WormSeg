import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np
import os

# paths
model_dir = "worm-segformer/checkpoint-xxx"  # replace with best checkpoint folder
image_path = "output_folder52_resized_inputs/val/frame_0001.png"
output_path = "prediction.png"

# processor + model
feature_extractor = SegformerImageProcessor(do_resize=True, size=224, do_normalize=True)
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
model.eval()

# load image
image = Image.open(image_path).convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt")

# predict
with torch.no_grad():
    outputs = model(**inputs)
preds = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()

# save binary mask
mask_img = Image.fromarray((preds * 255).astype(np.uint8))
mask_img.save(output_path)

print(f"Saved prediction to {output_path}")
