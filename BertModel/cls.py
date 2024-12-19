import torch.nn as nn

class CombinedClassifier(nn.Module):
    def __init__(self, combined_dim, hidden_dim, num_classes):
        super(CombinedClassifier, self).__init__()
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
