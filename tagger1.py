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
        words = torch.tensor([])

        # get the embedded vectors of each word and concat to a large vector
        for idx in words_idxs:
            # convert the index to tensor
            idx = torch.tensor([idx], dtype=torch.long)
            # get embedding vector of the word
            words = torch.cat((words, self.embed_layer(idx)), dim=1)

        x = F.tanh(self.layer1(words))
        out = F.softmax(self.layer2(x))

        return out


def train_model(train_set, model,  n_epochs, lr, device, word2index, label2index):
    model.to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(n_epochs):
        train(model, train_set, optimizer, criterion, e, word2index, label2index)


def train(model, train_set, optimizer, criterion, epoch, word2index, label2index):
    # num_labels = torch.tensor([len(label2index.keys())])
    num_labels = len(label2index.keys())
    running_loss = 0
    for i, data in enumerate(train_set):
        label, words = data
        words_idxs = []

        # get the indices of all the words
        for word in words:
            words_idxs.append(word2index[word])

        # get the label of the middle word
        label_idx = torch.tensor(label2index[label], dtype=torch.long)

        optimizer.zero_grad()

        # predict
        outputs = model(words_idxs)

        # create one hot vector of the label
        label_idx_vector = F.one_hot(label_idx, num_labels)
        # label_idx_vector.T[label_idx] = 1

        loss = criterion(outputs.view([-1]), label_idx_vector)
        loss.backwards()

        # backwards step
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] train loss:{running_loss}')
            running_loss = 0

    # return model


def evaluate(model, train_set, optimizer, criterion, epoch, word2index, label2index):
    running_loss = 0
    for i, data in enumerate(train_set):
        label, words = data
        words_idxs = []

        # get the indices of all the words
        for word in words:
            words_idxs.append(word2index[word])

        # get the label of the middle word
        label_idx = torch.tensor(label2index[label], dtype=torch.long)

        # predict
        outputs = model(words_idxs)
        loss = criterion(outputs, label_idx)
        loss.backwards()

        # backwards step
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] dev loss:{running_loss}')
            running_loss = 0

    # return model


def predict(test_set, model, device, words2index, index2label):
    model.to(device)
    model.eval()

    predicted_labels = []

    for i, data in enumerate(test_set):
        words = data
        words_idxs = []

        for word in words:
            words_idxs.append(words2index[word])

        # predict
        outputs = model(words_idxs)

        # get the index of the label
        index = torch.argmax(outputs)

        # ge the label from the index
        label = index2label[index]

        predicted_labels.append(label)

    return predicted_labels
