import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Tagger1Model(nn.Module):
    def __init__(self, vocab_size, embed_size, num_words, hidden_dim, out_dim):
        super(Tagger1Model, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_size)

        self.layer1 = nn.Linear(num_words * embed_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, words_idxs):
        words = self.embed_layer(words_idxs)
        x = torch.cat(words)
        x = F.tanh(self.layer1(x))
        out = F.softmax(self.layer2(x))

        return out


def train(train_set, dev_set, model, n_epochs, lr, device):
    model.to(device)


def predict(test_set, model, device):
    model.to(device)
    model.eval()
