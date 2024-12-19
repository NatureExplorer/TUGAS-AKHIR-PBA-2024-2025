from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from utils import prepare_data, collate_fn
from models import LSTMClassifier
import torch

# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 4
BATCH_SIZE = 32
EPOCHS = 5

# File paths
train_path = "train.csv"
test_path = "test.csv"
classes_path = "classes.txt"
glove_file_path = "glove.6B.100d.txt"

# Load data
train_dataset, test_dataset = prepare_data(train_path, test_path, classes_path, glove_file_path)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Model, optimizer, loss
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, y, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(X, lengths)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}")

# Evaluate
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X, y, lengths in test_loader:
        outputs = model(X, lengths)
        preds = outputs.argmax(dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
