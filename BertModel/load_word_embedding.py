import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Generate GloVe embedding for a text
def text_to_glove_embedding(text, embeddings, max_len, embedding_dim):
    tokens = text.split()[:max_len]
    embedding_matrix = np.zeros((max_len, embedding_dim))
    for i, word in enumerate(tokens):
        if word in embeddings:
            embedding_matrix[i] = embeddings[word]
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# Generate BERT embedding for a text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

# Combine GloVe and BERT embeddings
def combine_embeddings(text, glove_embeddings, bert_model, tokenizer, max_len=50, embedding_dim=100):
    glove_embedding = text_to_glove_embedding(text, glove_embeddings, max_len, embedding_dim)
    bert_embedding = get_bert_embedding(text)
    combined = torch.cat((glove_embedding.mean(dim=0), bert_embedding.squeeze(0)), dim=-1)  # Combine along feature axis
    return combined
