import warnings

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          get_scheduler)

warnings.filterwarnings("ignore")

# Get Dataset
raw_datasets = load_dataset("glue", "mrpc")

# Model Checkpoint
checkpoint = "bert-base-uncased"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Function to help tokenize with limited memory usage
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# Batched tokenization for faster tokenization
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Group data into batches with dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Need to remove unecessary columns from dataset to feed into dataloader
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Dataloader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Defining lr schedular

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_schedular = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps = 0,
    num_training_steps=num_training_steps
)


# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Loop
model.to(DEVICE)


# Progress Bar
progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):

    for batch in train_dataloader:
        # Convert to correct data format
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        # Feed to model
        outputs = model(**batch)

        # Loss
        loss = outputs.loss

        # Backwards
        loss.backward()

        optimizer.step()
        lr_schedular.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Evaluation Loop

metric = evaluate.load("glue", "mrpc")

model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())