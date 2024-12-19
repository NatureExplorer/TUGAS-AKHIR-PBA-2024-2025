import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # x: [batch_size, seq_length, embedding_dim], lengths: [batch_size]
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_input)
        out = self.fc(hn[-1])  # Ambil hidden state terakhir
        return out  # Langsung logits, softmax di luar model saat inferensi


class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, output_dim):
        super(TransformerClassifier, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads), num_layers
        )
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_length, embedding_dim] -> transpose to [seq_length, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Ambil mean di sepanjang seq_length
        x = self.fc(x)
        return x  # Langsung logits

from transformers import BertForSequenceClassification

class BERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BERTClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits  # Langsung logits


