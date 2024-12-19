import pandas as pd
import re
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset
import numpy as np

def load_data(train_path, test_path, classes_path):
    """Membaca dataset train dan test serta memuat kelas-kelas dari file classes.txt."""
    train_df = pd.read_csv(train_path, delimiter=",", header=None, names=["label", "title", "description"], quotechar='"')
    test_df = pd.read_csv(test_path, delimiter=",", header=None, names=["label", "title", "description"], quotechar='"')
    
    # Gabungkan title dan description untuk teks
    train_df["text"] = train_df["title"] + " " + train_df["description"]
    test_df["text"] = test_df["title"] + " " + test_df["description"]

    # Memuat kelas dari classes.txt
    with open(classes_path, "r") as f:
        class_names = f.readlines()
    class_names = [x.strip() for x in class_names]

    # Ubah label numerik menjadi nama kelas
    train_df["label"] = train_df["label"].apply(lambda x: class_names[x - 1])  # adjust class index to match 0-based index
    test_df["label"] = test_df["label"].apply(lambda x: class_names[x - 1])

    return train_df, test_df


def clean_text(text):
    """Membersihkan teks."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Hapus karakter spesial
    text = text.lower()
    return text


from torch.nn.utils.rnn import pad_sequence
from torch import nn
import numpy as np
import torch
from nltk.tokenize import word_tokenize

class AGNewsDataset(Dataset):
    """Dataset untuk PyTorch dengan format AG's News."""
    def __init__(self, texts, labels, word_vectors, class_names):
        # Tokenize and create embeddings for the texts
        self.texts = [
            torch.tensor(
                [word_vectors[word] for word in word_tokenize(t) if word in word_vectors],
                dtype=torch.float
            ) if any(word in word_vectors for word in word_tokenize(t)) else torch.zeros((1, word_vectors['the'].shape[0]))
            for t in texts
        ]

        # Convert string labels to numeric labels using the class_names mapping
        self.labels = torch.tensor(
            [class_names.index(label) if label in class_names else -1 for label in labels],
            dtype=torch.long
        )

        if (self.labels < 0).any():
            raise ValueError("Some labels are invalid or not found in class_names!")


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])  # Panjang sebelum padding
    texts = [text if len(text) > 0 else torch.zeros((1, text.shape[1])) for text in texts]
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.stack(labels), lengths



def load_glove_model(glove_file_path):
    """Memuat model GloVe dari file .txt."""
    glove_model = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) > 2:
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                glove_model[word] = vector
    return glove_model


def prepare_data(train_path, test_path, classes_path, glove_file_path):
    """Load, preprocess, and split data."""
    train_df, test_df = load_data(train_path, test_path, classes_path)

    # Preprocessing text
    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    # Load GloVe model
    glove_model = load_glove_model(glove_file_path)

    # Read class names
    class_names = open(classes_path).read().strip().splitlines()

    # Prepare datasets
    train_dataset = AGNewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), glove_model, class_names)
    test_dataset = AGNewsDataset(test_df["text"].tolist(), test_df["label"].tolist(), glove_model, class_names)

    return train_dataset, test_dataset

