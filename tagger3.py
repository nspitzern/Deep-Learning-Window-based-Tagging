import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from utils import save_model, draw_graphs, check_number


class Tagger3Model(nn.Module):
    def __init__(self, vocab_size, embed_size, num_words, hidden_dim, out_dim, prefix_size, suffix_size, is_pretrained=False, embeddings=None):
        super(Tagger3Model, self).__init__()
        self.num_words = num_words
        self.embed_size = embed_size

        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        if is_pretrained:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings).float())

        self.prefix_embed_layer = nn.Embedding(prefix_size, embed_size)
        self.suffix_embed_layer = nn.Embedding(suffix_size, embed_size)

        self.layer1 = nn.Linear(num_words * embed_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, words_idxs, prefix_idxs, suffix_idxs):

        # get the embedded vectors of each word and concat to a large vector
        x = self.embedding_layer(words_idxs).view(-1, self.num_words * self.embed_size)

        # get the embedded vectors of each word and concat to a large vector
        prefix = self.prefix_embed_layer(prefix_idxs).view(-1, self.num_words * self.embed_size)

        # get the embedded vectors of each word and concat to a large vector
        suffix = self.suffix_embed_layer(suffix_idxs).view(-1, self.num_words * self.embed_size)

        x = prefix + x + suffix

        x = torch.tanh(self.layer1(x))
        x = self.dropout(x)
        out = self.softmax(self.layer2(x))

        return out


def train_model(train_set, dev_set, model,  n_epochs, lr, device, index2word, word2index, index2label, is_pos=False):
    model.to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracy = []
    dev_losses = []
    dev_accuracy = []

    for e in range(n_epochs):
        train_loss = train(model, train_set, optimizer, criterion, device)
        _, train_acc = evaluate(model, train_set, criterion, device, index2label, is_pos)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        dev_loss, accuracy = evaluate(model, dev_set, criterion, device, index2label, is_pos)
        dev_losses.append(dev_loss)
        dev_accuracy.append(accuracy)

        print(f'[{e + 1}/{n_epochs}] train loss: {train_loss}, train accuracy: {train_acc}%,'
              f' dev loss: {dev_loss}, dev accuracy: {accuracy}%')

    save_model(model, train_losses, train_accuracy, dev_losses, dev_accuracy, '.')

    # draw graphs of loss and accuracy history
    draw_graphs(train_losses, dev_losses, n_epochs, 'Loss History', 'Train Loss', 'Validation Loss')
    draw_graphs(train_accuracy, dev_accuracy, n_epochs, 'Accuracy History', 'Train Accuracy', 'Validation Accuracy')


def train(model, train_set, optimizer, criterion, device):
    running_loss = 0
    for i, data in enumerate(train_set):
        labels_batch, words_batch, prefix_idxs, suffix_idxs = data

        words_batch = torch.stack(words_batch, dim=1)
        prefix_idxs = torch.stack(prefix_idxs, dim=1)
        suffix_idxs = torch.stack(suffix_idxs, dim=1)

        words_batch = words_batch.to(device)
        labels_batch = labels_batch.to(device)
        prefix_idxs = prefix_idxs.to(device)
        suffix_idxs = suffix_idxs.to(device)

        optimizer.zero_grad()

        # predict
        outputs = model(words_batch, prefix_idxs, suffix_idxs)

        loss = criterion(outputs.squeeze(), labels_batch)
        loss.backward()

        # backwards step
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_set.dataset)


def evaluate(model, dev_set, criterion, device, index2label, is_pos):
    running_loss = 0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(dev_set):
        labels_batch, words_batch, prefix_idxs, suffix_idxs = data

        words_batch = torch.stack(words_batch, dim=1)
        prefix_idxs = torch.stack(prefix_idxs, dim=1)
        suffix_idxs = torch.stack(suffix_idxs, dim=1)

        words_batch = words_batch.to(device)
        labels_batch = labels_batch.to(device)
        prefix_idxs = prefix_idxs.to(device)
        suffix_idxs = suffix_idxs.to(device)

        # predict
        outputs = model(words_batch, prefix_idxs, suffix_idxs)

        loss = criterion(outputs.squeeze(), labels_batch)

        running_loss += loss.item()

        predictions = torch.argmax(outputs.data, dim=1)

        if is_pos:
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)
        else:
            for prediction, real_label in zip(predictions, labels_batch):
                # count how many labels were in this batch
                total += 1

                # check if the prediction in like the real label
                if prediction == real_label:
                    # if both are 'O' skip it because there are many 'O's (don't count it)
                    if index2label[int(prediction)] == 'O':
                        total -= 1
                    else:
                        # otherwise count the correct results
                        correct += 1

    return running_loss / len(dev_set.dataset), round(100 * correct / total, 3)


def predict(test_set, model, device, index2label):
    model.to(device)
    model.eval()

    predicted_labels = []

    for i, data in enumerate(test_set):
        words_batch, prefix_batch, suffix_batch = data
        words_batch = torch.stack(words_batch, dim=1)
        prefix_batch = torch.stack(prefix_batch, dim=1)
        suffix_batch = torch.stack(suffix_batch, dim=1)

        words_batch = words_batch.to(device)
        prefix_batch = prefix_batch.to(device)
        suffix_batch = suffix_batch.to(device)

        # predict
        outputs = model(words_batch, prefix_batch, suffix_batch)

        # get the index of the label
        index = torch.argmax(outputs)

        # ge the label from the index
        label = index2label.get(index.item(), 'UNSEEN')

        predicted_labels.append(label)

    return predicted_labels


def convert_dataset_to_index(dataset, word2index, label2index, prefix2index, suffix2index, pretrained=False, is_test=False):
    for i in range(len(dataset)):
        # get current sample
        if not is_test:
            pos, words = dataset[i]
        else:
            words = dataset[i]

        suffix = []
        prefix = []
        # go over the words in the window
        for j in range(len(words)):
            if pretrained:
                words[j] = check_number(words[j], word2index.keys())
            # for each word, check if word is in the training set. if not change to unknown
            word = words[j] if words[j] in word2index else 'UUUNKKK'
            # get hte prefix and suffix of the word
            prefix.append(prefix2index[word[:3]])
            suffix.append(suffix2index[word[-3:]])
            # convert word to index. if the word was not seen - convert to unseen letter
            if not is_test:
                dataset[i][1][j] = word2index[word]
            else:
                dataset[i][j] = word2index[word]
        # change the tag to index
        dataset[i] = list(dataset[i])
        dataset[i] = list(dataset[i])
        if not is_test:
            dataset[i][0] = label2index.get(pos, label2index['<UNSEEN>'])
        else:
            dataset[i] = [dataset[i]]
        # add the prefixes and suffixes (indices) of the words in the window
        dataset[i].append(prefix)
        dataset[i].append(suffix)
        dataset[i] = tuple(dataset[i])

    return dataset
