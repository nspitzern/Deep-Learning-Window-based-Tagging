import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Tagger1Model(nn.Module):
    def __init__(self, batch_size, vocab_size, embed_size, num_words, hidden_dim, out_dim):
        super(Tagger1Model, self).__init__()
        self.batch_size = batch_size
        self.embed_layer = nn.Embedding(vocab_size, embed_size)

        self.layer1 = nn.Linear(num_words * embed_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, words_idxs):
        # get the embedded vectors of each word and concat to a large vector
        # words = torch.stack(words_idxs)
        words = words_idxs
        x = self.embed_layer(words).view((self.batch_size, -1))

        x = torch.tanh(self.layer1(x))
        out = self.softmax(self.layer2(x))

        return out


def train_model(train_set, dev_set, model,  n_epochs, lr, device, word2index, label2index):
    model.to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    dev_losses = []
    dev_accuracy = []
    for e in range(n_epochs):
        train_loss = train(model, train_set, optimizer, criterion, e, device, word2index, label2index)
        train_losses.append(train_loss)

        dev_loss, accuracy = evaluate(model, dev_set, criterion, e, device,word2index, label2index)
        dev_losses.append(dev_loss)
        dev_accuracy.append(accuracy)

        print(f'[{e}/{n_epochs}] dev loss: {dev_loss}, accuracy: {accuracy}')


def train(model, train_set, optimizer, criterion, epoch, device, word2index, label2index):
    running_loss = 0
    for i, data in enumerate(train_set):
        labels_batch, words_batch = data

        words_batch = torch.stack(words_batch)

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

        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] train loss:{running_loss}')
            running_loss = 0

    return running_loss


def evaluate(model, dev_set, criterion, epoch, device, word2index, label2index):
    running_loss = 0
    accuracy = 0
    total_num = 0
    for i, data in enumerate(dev_set):
        labels_batch, words_batch = data

        words_batch = torch.stack(words_batch)

        words_batch = words_batch.to(device)
        labels_batch = labels_batch.to(device)

        # predict
        outputs = model(words_batch)

        loss = criterion(outputs.squeeze(), labels_batch)

        running_loss += loss.item()

        predictions = torch.argmax(outputs)

        accuracy += sum(predictions == labels_batch)
        total_num += len(labels_batch)

        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] dev loss: {running_loss}')
            running_loss = 0

    return running_loss, accuracy / total_num


def predict(test_set, model, device, words2index, index2label):
    model.to(device)
    model.eval()

    predicted_labels = []

    for i, data in enumerate(test_set):
        words_batch = data
        words_batch = torch.stack(words_batch)

        words_batch = words_batch.to(device)

        # predict
        outputs = model(words_batch)

        # get the index of the label
        index = torch.argmax(outputs)

        # ge the label from the index
        label = index2label[index]

        predicted_labels.append(label)

    return predicted_labels
