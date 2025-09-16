from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test-output",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
)

print("TrainingArguments created successfully!")
