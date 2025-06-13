from pali_dataset import ViVQAPaLIDataset
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from pali import PaLI

BATCH_SIZE = 1
LEARNING_RATE = 2e-15
EPOCHS = 50

train_dataset = ViVQAPaLIDataset(
    json_path="/root/projects/exp1/data/vivqa/vqa/train_en.json",
    image_dir="/root/projects/exp1/data/vivqa/images/train",
)

test_dataset = ViVQAPaLIDataset(
    json_path="/root/projects/exp1/data/vivqa/vqa/test_en.json",
    image_dir="/root/projects/exp1/data/vivqa/images/test",
)

train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)

test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model =PaLI(device=device).to(device, non_blocking=True)

model.train()

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    for batch in train_data_loader:
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(pixel_values, input_ids, attention_mask, labels)
        loss = output.loss

        loss.backward()
        optimizer.step()
        
