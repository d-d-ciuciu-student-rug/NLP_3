import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

# load data
train_df = pd.read_json("hf://datasets/sh0416/ag_news/train.jsonl", lines=True)
test_df = pd.read_json("hf://datasets/sh0416/ag_news/test.jsonl", lines=True)

train_df["text"] = (
    train_df["title"].fillna("") + " " + train_df["description"].fillna("")
).str.strip()
test_df["text"] = (
    test_df["title"].fillna("") + " " + test_df["description"].fillna("")
).str.strip()


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = AGNewsDataset(
    train_df["text"].tolist(), train_df["label"].tolist(), tokenizer
)
test_dataset = AGNewsDataset(
    test_df["text"].tolist(), test_df["label"].tolist(), tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

# training
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

# evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# metrics
acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {acc:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("Confusion Matrix:")
print(cm)
