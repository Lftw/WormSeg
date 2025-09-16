import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, TrainingArguments, Trainer
from dataset import WormDataset

# Ensure GPU is used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# label mapping
id2label = {0: "background", 1: "worm"}
label2id = {"background": 0, "worm": 1}

# preprocessing
feature_extractor = SegformerImageProcessor(do_resize=True, size=224, do_normalize=True)

# datasets
train_dataset = WormDataset("../output_folder52_resized_inputs/train", "../output_folder52_resized_masks/train", feature_extractor)
val_dataset = WormDataset("../output_folder52_resized_inputs/val", "../output_folder52_resized_masks/val", feature_extractor)

# model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # 👈 important fix
)
model.to(device)


# training config
args = TrainingArguments(
    output_dir="worm-segformer",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",  # turn off wandb/hf logging unless you want it
)

# trainer

# Custom Trainer to move data to GPU in training step
from transformers import Trainer

class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        # Move all tensors in inputs to device
        device = model.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        return super().training_step(model, inputs)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and save best checkpoint
train_result = trainer.train()
trainer.save_model("worm-segformer/best-checkpoint")
