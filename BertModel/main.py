from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from cls import *
from load_word_embedding import *
import pandas as pd
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Dataset class
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, glove_embeddings, bert_model, tokenizer, max_len, embedding_dim):
        self.texts = texts
        self.labels = labels
        self.glove_embeddings = glove_embeddings
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        combined_embedding = combine_embeddings(
            text, self.glove_embeddings, self.bert_model, self.tokenizer, self.max_len, self.embedding_dim
        )
        return combined_embedding, torch.tensor(label, dtype=torch.long)


train = pd.read_csv("/kaggle/input/ta-nlp/ag_news_csv/train.csv",header=None,names=["label","title","description"])
test = pd.read_csv('/kaggle/input/ta-nlp/ag_news_csv/test.csv',header = None,names=["label","title","description"])

train['text'] = train['title'] + " " + train['description']
test['text'] = test['title'] + " " + test['description']

train['label'] = train['label'] - 1
test['label'] = test['label'] -1

x_train = train['text']
y_train = train['label']

x_test = test['text']
y_test = test['label']

glove_embeddings = load_glove_embeddings("/kaggle/input/word-embedding-glove/glove.6B.100d.txt")

# Prepare data
train_dataset = AGNewsDataset(x_train, y_train, glove_embeddings, bert_model, tokenizer, max_len=50, embedding_dim=100)
test_dataset = AGNewsDataset(x_test, y_test, glove_embeddings, bert_model, tokenizer, max_len=50, embedding_dim=100)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedClassifier(combined_dim=768 + 100, hidden_dim=256, num_classes=4)
model.to(device)

# Training loop
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    return accuracy, f1

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training and evaluation
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    accuracy, f1 = evaluate_model(model, test_loader, device)
    print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")
