import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaModel, AdamW
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
from tqdm import tqdm




class CodeCommentDataset(Dataset):
    def __init__(self, comments, snippets, labels1, labels2, tokenizer, max_length=512):
        self.comments = comments
        self.snippets = snippets
        self.labels1 = labels1
        self.labels2 = labels2
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        snippet = self.snippets[idx]
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]

        input_text = comment + " " + snippet

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels1': torch.tensor(label1, dtype=torch.long),
            'labels2': torch.tensor(label2, dtype=torch.long)
        }


class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_classes1=2, num_classes2=2):
        super(MultiTaskModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier1 = nn.Linear(self.roberta.config.hidden_size, num_classes1)
        self.classifier2 = nn.Linear(self.roberta.config.hidden_size, num_classes2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation

        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)

        return logits1, logits2


def compute_loss(logits1, logits2, labels1, labels2, weight1=1.0, weight2=1.0):
    loss_fn = nn.CrossEntropyLoss()
    loss1 = loss_fn(logits1, labels1) * weight1
    loss2 = loss_fn(logits2, labels2) * weight2
    return loss1+loss2

def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, precision, recall, f1

# Load your JSON data
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    comments = [item['comment'] for item in data]
    snippets = [item['snippet'] for item in data]
    labels1 = [item['task_label'] for item in data]
    labels2 = [item['label'] for item in data]

    return comments, snippets, labels1, labels2


# Hyperparameters
model_name = 'microsoft/codebert-base'
batch_size = 32
learning_rate = 2e-5
num_epochs = 20

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Load data
comments, snippets, labels1, labels2 = load_data('./train_data/mt_data.json')

# Create Dataset
dataset = CodeCommentDataset(comments, snippets, labels1, labels2, tokenizer)

# Split dataset into train, validate, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModel(model_name).to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Scheduler for dynamic learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels1 = batch['labels1'].to(device)
        labels2 = batch['labels2'].to(device)

        logits1, logits2 = model(input_ids, attention_mask)
        loss = compute_loss(logits1, logits2, labels1, labels2, weight1=1.0, weight2=1.0)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_train_loss / len(train_dataloader)}')

    if (epoch+1) % 5 == 0:
        # Validation loop
        model.eval()
        total_val_loss = 0
        all_preds1, all_preds2, all_labels1, all_labels2 = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", unit="batch"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels1 = batch['labels1'].to(device)
                labels2 = batch['labels2'].to(device)

                logits1, logits2 = model(input_ids, attention_mask)
                loss = compute_loss(logits1, logits2, labels1, labels2, weight1=1.0, weight2=1.0)
                total_val_loss += loss.item()

                preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
                preds2 = torch.argmax(logits2, dim=1).cpu().numpy()
                all_preds1.extend(preds1)
                all_preds2.extend(preds2)
                all_labels1.extend(labels1.cpu().numpy())
                all_labels2.extend(labels2.cpu().numpy())

        val_accuracy1, val_precision1, val_recall1, val_f1_1 = compute_metrics(all_labels1, all_preds1)
        val_accuracy2, val_precision2, val_recall2, val_f1_2 = compute_metrics(all_labels2, all_preds2)

        print(f'Validation Loss: {total_val_loss / len(val_dataloader)}')
        print(f"Task 1 - Accuracy: {val_accuracy1}, Precision: {val_precision1}, Recall: {val_recall1}, F1: {val_f1_1}")
        print(f"Task 2 - Accuracy: {val_accuracy2}, Precision: {val_precision2}, Recall: {val_recall2}, F1: {val_f1_2}")

    scheduler.step()  # Update learning rate

# Testing loop
model.eval()
test_preds1, test_preds2, test_labels1, test_labels2 = [], [], [], []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels1 = batch['labels1'].to(device)
        labels2 = batch['labels2'].to(device)

        logits1, logits2 = model(input_ids, attention_mask)
        preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
        preds2 = torch.argmax(logits2, dim=1).cpu().numpy()
        test_preds1.extend(preds1)
        test_preds2.extend(preds2)
        test_labels1.extend(labels1.cpu().numpy())
        test_labels2.extend(labels2.cpu().numpy())

test_accuracy1, test_precision1, test_recall1, test_f1_1 = compute_metrics(test_labels1, test_preds1)
test_accuracy2, test_precision2, test_recall2, test_f1_2 = compute_metrics(test_labels2, test_preds2)

print(f"Test Task 1 - Accuracy: {test_accuracy1}, Precision: {test_precision1}, Recall: {test_recall1}, F1: {test_f1_1}")
print(f"Test Task 2 - Accuracy: {test_accuracy2}, Precision: {test_precision2}, Recall: {test_recall2}, F1: {test_f1_2}")

# Save model
torch.save(model.state_dict(), './model/multi_task_codebert_model.pth')
tokenizer.save_pretrained('./model/multi_task_codebert_model')

# Load model
# model = MultiTaskModel(model_name)
# model.load_state_dict(torch.load('./model/multi_task_codebert_model.pth'))
# model.to(device)

