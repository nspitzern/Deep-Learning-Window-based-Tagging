import numpy as np
import matplotlib.pyplot as plt
import torch


def create_word_vec_dict():
    vecs = np.loadtxt("pretrained vectors.txt")
    with open("words.txt", 'r') as words_file:
        words = words_file.read().splitlines()
    words2vecs = {'<s>': torch.zeros(vecs.shape[1], dtype=torch.float), '<e>': torch.zeros(vecs.shape[1], dtype=torch.float)}
    words2inx = {'<s>': 0, '<e>': 1}
    # words2vecs = {}
    # words2inx = {}
    i, j = 0, len(words2vecs)
    for word in words:
        if word not in words2vecs:
            words2inx[word] = j
            vec = vecs[i]
            words2vecs[word] = vec
            j += 1
        i += 1
    return words2vecs, words2inx


def create_char_inx_dict():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    char2index = {'<s>': 0, '<e>': 1, '<UNSEEN>': 2}
    index2char = {0: '<s>', 1: '<e>', 2: '<UNSEEN>'}
    for inx, char in enumerate(alphabet):
        char2index[char] = inx + 3
        
    return char2index, index2char


def parse_NER(file_path, window_size):
    # initialize dictionaries
    word2index = {'<s>': 0, '<e>': 1, 'UUUNKKK': 2}
    index2word = {0: '<s>', 1: '<e>', 2: 'UUUNKKK'}
    label2index = {'<START>': 0, '<END>': 1, '<UNSEEN>': 2}
    index2label = {0: '<START>', 1: '<END>', 2: '<UNSEEN>'}

    with open(file_path, 'r', encoding='utf-8') as f:
        word_index = len(word2index)
        label_index = len(label2index)
        dataset = []

        # split into sentences (separated by blank rows)
        sentences = f.read().split('\n\n')
        ignored = ['<s>', '<e>', '-docstart-']

        for sentence in sentences:
            if sentence == '' or sentence == '\n':
                continue
            # add special words of start and end of sentence with special labels (<S> = START, <E> = END)
            sentence = '<s>\tSTART\n' * window_size + sentence + '\n<e>\tEND' * window_size
            words = sentence.split('\n')

            # go over the words (not including the start and end words)
            for i in range(window_size, len(words) - window_size):
                # for each word split into word and label
                word, ner = words[i].split('\t')
                word = word.lower()
                dataset.append((ner, [w.split('\t')[0].lower() for w in words[i - window_size: i + window_size + 1]]))

                # keep track of word and index
                if word not in word2index:
                    word2index[word] = word_index
                    index2word[word_index] = word
                    word_index += 1

                # keep track of label and index
                if ner not in label2index:
                    label2index[ner] = label_index
                    index2label[label_index] = ner
                    label_index += 1

    return dataset, word2index, index2word, label2index, index2label


def parse_POS(file_path, window_size):
    # initialize dictionaries
    word2index = {'<s>': 0, '<e>': 1, 'UUUNKKK': 2}
    index2word = {0: '<s>', 1: '<e>', 2: 'UUUNKKK'}
    label2index = {'<START>': 0, '<END>': 1, '<UNSEEN>': 2}
    index2label = {0: '<START>', 1: '<END>', 2: '<UNSEEN>'}

    with open(file_path, 'r', encoding='utf-8') as f:
        word_index = len(word2index)
        label_index = len(label2index)
        dataset = []

        # split into sentences (separated by blank rows)
        sentences = f.read().split('\n\n')

        for sentence in sentences:
            if sentence == '' or sentence == '\n':
                continue
            # add special words of start and end of sentence with special labels (<S> = START, <E> = END)
            sentence = '<s> START\n' * window_size + sentence + '\n<e> END' * window_size
            words = sentence.split('\n')

            # go over the words (not including the start and end words)
            for i in range(window_size, len(words) - window_size):
                # for each word split into word and label
                word, pos = words[i].split(' ')
                word = word.lower()

                dataset.append((pos, [w.split(' ')[0].lower() for w in words[i - window_size: i + window_size + 1]]))
                # keep track of word and index
                if word not in word2index:
                    word2index[word] = word_index
                    index2word[word_index] = word
                    word_index += 1

                # keep track of label and index
                if pos not in label2index:
                    label2index[pos] = label_index
                    index2label[label_index] = pos
                    label_index += 1

    return dataset, word2index, index2word, label2index, index2label


def check_number(word, vocab):
    # word is a number (positive, negative, whole, float)
    if all(c.isdigit() or c == '.' or c == '-' or c == '+' for c in word):
        new_word = ''

        for c in word:
            new_word += 'DG' if c.isdigit() else c

        if new_word in vocab:
            return new_word
        else:
            return 'NNNUMMM'

    # word is of form '###,#####'
    elif all(ch.isdigit() or ch == ',' for ch in word) and any(ch.isdigit() for ch in word):
        return "NNNUMMM"
    return word


def convert_dataset_to_index(dataset, word2index, label2index, pretrained=False):
    for i in range(len(dataset)):
        # get current sample
        pos, words = dataset[i]
        # go over the words in the window
        for j in range(len(words)):
            if pretrained:
                words[j] = check_number(words[j], word2index.keys())
            # convert word to index. if the word was not seen - convert to unseen letter
            dataset[i][1][j] = word2index.get(words[j], word2index['UUUNKKK'])
        # change the tag to index
        dataset[i] = list(dataset[i])
        dataset[i][0] = label2index.get(pos, label2index['<UNSEEN>'])
        dataset[i] = tuple(dataset[i])

    return dataset


def convert_to_sub_words(word2index, sub_word_size=3):
    prefix_dict = dict()
    suffix_dict = dict()

    prefix_idx = 0
    suffix_idx = 0

    for word, index in word2index.items():
        prefix = word[:sub_word_size]
        suffix = word[-sub_word_size:]

        if prefix not in prefix_dict:
            prefix_dict[prefix] = prefix_idx
            prefix_idx += 1
        if suffix not in suffix_dict:
            suffix_dict[suffix] = suffix_idx
            suffix_idx += 1

    return prefix_dict, suffix_dict


def parse_test_file(file_path, window_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = []

        # split into sentences (separated by blank rows)
        sentences = f.read().split('\n\n')

        for sentence in sentences:
            if sentence == '' or sentence == '\n':
                continue
            # add special words of start and end of sentence with special labels (<S> = START, <E> = END)
            sentence = '<s>\n' * window_size + sentence + '\n<e>' * window_size
            words = sentence.split('\n')

            # go over the words (not including the start and end words)
            for i in range(window_size, len(words) - window_size):
                # insert to the dataset a tuple of label and 5 words when the label is of the middle word
                dataset.append(list(word.lower() for word in words[i - window_size: i + window_size + 1]))

    return dataset


def export_predictions(predictions, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(predictions))


def save_model(model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history, path):
    torch.save(model.state_dict(), f'{path}/model.path')
    torch.save(train_loss_history, f'{path}/train_loss_history.path')
    torch.save(train_accuracy_history, f'{path}/train_accuracy_history.path')
    torch.save(dev_loss_history, f'{path}/dev_loss_history.path')
    torch.save(dev_accuracy_history, f'{path}/dev_accuracy_history.path')


def load_model(model, model_path, train_loss_history_path, train_accuracy_history_path, dev_loss_history_path, dev_accuracy_history_path):
    model.load_state_dict(torch.load(model_path))
    train_loss_history = torch.load(train_loss_history_path)
    train_accuracy_history = torch.load(train_accuracy_history_path)
    dev_loss_history = torch.load(dev_loss_history_path)
    dev_accuracy_history = torch.load(dev_accuracy_history_path)
    return model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history


def draw_graphs(train_history, dev_history, n_epochs, plot_title, train_title, dev_title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(plot_title)
    x = torch.arange(n_epochs) + 1
    ax1.set_title(train_title)
    ax1.plot(x, train_history)
    ax2.set_title(dev_title)
    ax2.plot(x, dev_history)

    plt.show()

