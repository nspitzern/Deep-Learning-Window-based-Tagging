import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class Tagger1Model(nn.Module):
    def __init__(self, vocab_size, embed_size, num_words, hidden_dim, out_dim):
        super(Tagger1Model, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_size)

        self.layer1 = nn.Linear(num_words * embed_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, words_idxs):
        words = torch.empty()

        # get the embedded vectors of each word and concat to a large vector
        for idx in words_idxs:
            # convert the index to tensor
            idx = torch.tensor([idx], dtype=torch.long)
            # get embedding vector of the word
            torch.cat((words, self.embed_layer(idx)))

        x = F.tanh(self.layer1(words))
        out = F.softmax(self.layer2(x))

        return out


def train_model(train_set, model,  n_epochs, lr, device, words2index, label2index):
    model.to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(n_epochs):
        train(model, train_set, optimizer, criterion, e, words2index, label2index)

def train(model, train_set, optimizer, criterion, epoch, words2index, label2index):
    running_loss = 0
    for i, data in enumerate(train_set):
        label, words = data
        words_idxs = []

        # convert the
        for word in zip(words, label):
            words_idxs.append(words2index[word])
        labels_idx = torch.tensor(label2index[label], dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(words_idxs)
        loss = criterion(outputs, labels_idx)
        loss.backwards()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss:{running_loss}')
            running_loss = 0

    return model


def evaluate(model, train_set, optimizer, criterion, epoch):
    running_loss = 0
    for i, data in enumerate(train_set):
        labels, words = data
        words_idxs = []
        labels_idxs = []

        for label, word in zip(words, labels):
            words_idxs.append()

        outputs = model(words_idxs)
        loss = criterion(outputs, labels_idxs)
        loss.backwards()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss:{running_loss}')
            running_loss = 0

    return model


def predict(test_set, model, device):
    model.to(device)
    model.eval()

    predicted_labels = []

    for i, data in enumerate(test_set):
        labels, words = data
        words_idxs = []
        labels_idxs = []

        for label, word in zip(words, labels):
            words_idxs.append()

        outputs = model(words_idxs)
        index = torch.argmax(outputs)
        label =

    return predicted_labels
