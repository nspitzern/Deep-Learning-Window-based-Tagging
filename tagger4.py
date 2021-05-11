import torch
import torch.nn as nn
import torch.optim as optim

from utils import save_model, draw_graphs, check_number


class Tagger4Model(nn.Module):
    def __init__(self, vocab_size, embed_size, c_embed_size, num_words, hidden_dim, out_dim, chars_size,
                 total_chars_in_corpus, is_pretrained=False, embeddings=None):
        super(Tagger4Model, self).__init__()
        self.num_words = num_words
        self.embed_size = embed_size
        self.c_embed_size = c_embed_size
        self.chars_size = chars_size

        self.char_embed_layer = nn.Embedding(total_chars_in_corpus, c_embed_size)
        self.conv = nn.Conv1d(self.c_embed_size, self.c_embed_size, kernel_size=3, stride=1, padding=2)
        self.pooling = nn.MaxPool1d(self.chars_size)

        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        if is_pretrained:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings).float())

        self.layer1 = nn.Linear(num_words * embed_size + c_embed_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, words_idxs, chars_idxs):
        
        # get the embedded vectors of each char and concat to a large vector
        chars = self.char_embed_layer(chars_idxs)
        chars = self.dropout(chars)
        chars = self.conv(chars)
        chars = self.pooling(chars).view(-1, self.c_embed_size)
        
        # get the embedded vectors of each word and concat to a large vector
        x = self.embedding_layer(words_idxs).view(-1, self.num_words * self.embed_size)

        x = torch.cat((x, chars), dim=1)

        x = torch.tanh(self.layer1(x))
        out = self.softmax(self.layer2(x))

        return out


def train_model(train_set, dev_set, model,  n_epochs, lr, device, index2word, word2index, index2label, is_pos=False):
    model.to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.NLLLoss()

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
        labels_batch, words_batch, chars_idxs = data

        words_batch = torch.stack(words_batch, dim=1)
        chars_idxs = torch.stack(chars_idxs, dim=1)

        words_batch = words_batch.to(device)
        labels_batch = labels_batch.to(device)
        chars_idxs = chars_idxs.to(device)

        optimizer.zero_grad()

        # predict
        outputs = model(words_batch, chars_idxs)

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
        labels_batch, words_batch, chars_idxs = data

        words_batch = torch.stack(words_batch, dim=1)
        chars_idxs = torch.stack(chars_idxs, dim=1)

        words_batch = words_batch.to(device)
        labels_batch = labels_batch.to(device)
        chars_idxs = chars_idxs.to(device)

        # predict
        outputs = model(words_batch, chars_idxs)

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
        words_batch, chars_idxs = data

        words_batch = torch.stack(words_batch, dim=1)
        chars_idxs = torch.stack(chars_idxs, dim=1)

        words_batch = words_batch.to(device)
        chars_idxs = chars_idxs.to(device)

        # predict
        outputs = model(words_batch, chars_idxs)

        # get the index of the label
        index = torch.argmax(outputs)

        # ge the label from the index
        label = index2label.get(index.item(), 'UNSEEN')

        predicted_labels.append(label)

    return predicted_labels


def convert_dataset_to_index(dataset, word2index, label2index, char2index, max_word_size, pretrained=False, is_test=False):
    for i in range(len(dataset)):
        # get current sample
        if not is_test:
            pos, words = dataset[i]
        else:
            words = dataset[i]

        word_chars = []
        curr_word = words[len(words) // 2]
        # go over the words in the window
        for j in range(len(words)):
            if pretrained:
                words[j] = check_number(words[j], word2index.keys())
            # for each word, check if word is in the training set. if not change to unknown
            word = words[j] if words[j] in word2index else 'UUUNKKK'
            # convert word to index. if the word was not seen - convert to unseen letter
            if not is_test:
                dataset[i][1][j] = word2index[word]
            else:
                dataset[i][j] = word2index[word]

        # get the chars of the current word
        curr_word = _add_padding(curr_word, max_word_size)

        for k in range(len(curr_word)):
            word_chars.append(char2index.get(curr_word[k], char2index['<UNSEEN>']))

        # change the tag to index
        dataset[i] = list(dataset[i])
        dataset[i] = list(dataset[i])
        if not is_test:
            dataset[i][0] = label2index.get(pos, label2index['<UNSEEN>'])
        else:
            dataset[i] = [dataset[i]]
        # add chars of the words in the window
        dataset[i].append(word_chars)
        dataset[i] = tuple(dataset[i])

    return dataset


def _add_padding(word, max_word_size):
    # separate into characters
    word = list(word)
    # check the length of the word
    if len(word) < max_word_size:
        # if the word is smaller than the max_size - we need to pad
        pad_size = (max_word_size - len(word)) // 2
        word = ['<s>'] * pad_size + word + ['<e>'] * pad_size

        if len(word) % 2 != 0:
            word = ['<s>'] + word

    word = word[:max_word_size]

    return word
