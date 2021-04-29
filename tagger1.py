import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import save_model


class Tagger1Model(nn.Module):
    def __init__(self, batch_size, vocab_size, embed_size, num_words, hidden_dim, out_dim):
        super(Tagger1Model, self).__init__()
        self.batch_size = batch_size
        self.embed_layer = nn.Embedding(vocab_size, embed_size)

        self.layer1 = nn.Linear(num_words * embed_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, words_idxs):
        # get the embedded vectors of each word and concat to a large vector
        x = self.embed_layer(words_idxs).view((self.batch_size, -1))

        x = torch.tanh(self.layer1(x))
        # x = self.dropout(x)
        out = self.softmax(self.layer2(x))

        return out


def train_model(train_set, dev_set, model,  n_epochs, lr, device, word2index, label2index, is_pos=False):
    model.to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracy = []
    dev_losses = []
    dev_accuracy = []

    for e in range(n_epochs):
        train_loss = train(model, train_set, optimizer, criterion, device)
        _, train_acc = evaluate(model, train_set, criterion, device, is_pos)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        dev_loss, accuracy = evaluate(model, dev_set, criterion, device, is_pos)
        dev_losses.append(dev_loss)
        dev_accuracy.append(accuracy)

        print(f'[{e + 1}/{n_epochs}] train loss: {train_loss}, train accuracy: {train_acc}%,'
              f' dev loss: {dev_loss}, dev accuracy: {accuracy}%')

    save_model(model, train_losses, train_accuracy, dev_losses, dev_accuracy, '.')


def train(model, train_set, optimizer, criterion, device):
    running_loss = 0
    for i, data in enumerate(train_set):
        labels_batch, words_batch = data

        words_batch = torch.stack(words_batch, dim=1)

        words_batch = words_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        # predict
        outputs = model(words_batch)

        loss = criterion(outputs.squeeze(), labels_batch)
        loss.backward()

        # backwards step
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_set.dataset)


def evaluate(model, dev_set, criterion, device, is_pos):
    running_loss = 0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(dev_set):
        labels_batch, words_batch = data

        words_batch = torch.stack(words_batch, dim=1)

        words_batch = words_batch.to(device)
        labels_batch = labels_batch.to(device)

        # predict
        outputs = model(words_batch)

        loss = criterion(outputs.squeeze(), labels_batch)
        # loss = F.nll_loss(outputs.squeeze(), labels_batch)

        running_loss += loss.item()

        predictions = torch.argmax(outputs.data, dim=1)

        if is_pos:
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)
        else:
            pass

    return running_loss / len(dev_set.dataset), round(100 * correct / total, 3)


def predict(test_set, model, device, words2index, index2label):
    model.to(device)
    model.eval()

    predicted_labels = []

    for i, data in enumerate(test_set):
        words_batch = data
        words_batch = torch.stack(words_batch, dim=1)

        words_batch = words_batch.to(device)

        # predict
        outputs = model(words_batch)

        # get the index of the label
        index = torch.argmax(outputs)

        # ge the label from the index
        label = index2label[index]

        predicted_labels.append(label)

    return predicted_labels
