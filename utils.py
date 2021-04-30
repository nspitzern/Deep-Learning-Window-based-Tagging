import numpy as np
import torch


def create_word_vec_dict():
    vecs = np.loadtxt("pretrained vectors.txt")
    with open("words.txt", 'r') as words_file:
        words = words_file.read().splitlines()
    words2vecs = dict()
    words2inx = dict()
    i, j = 0, 0
    for word in words:
        if word not in words2vecs:
            words2inx[word] = j
            vec = vecs[i]
            words2vecs[word] = vec
            j += 1
        i += 1
    return words2vecs, words2inx


def parse_NER(file_path, window_size):
    # initialize dictionaries
    word2index = {'<S>': 0, '<E>': 1, '<U>': 2}
    index2word = {0: '<S>', 1: '<E>', 2: '<U>'}
    label2index = {'<START>': 0, '<END>': 1, '<UNSEEN>': 2}
    index2label = {0: '<START>', 1: '<END>', 2: '<UNSEEN>'}

    with open(file_path, 'r', encoding='utf-8') as f:
        word_index = 2
        label_index = 2
        dataset = []

        # split into sentences (separated by blank rows)
        sentences = f.read().split('\n\n')

        for sentence in sentences:
            if sentence == '' or sentence == '\n':
                continue
            # add special words of start and end of sentence with special labels (<S> = START, <E> = END)
            sentence = '<S>\tSTART\n<S>\tSTART\n' + sentence + '\n<E>\tEND\n<E>\tEND'
            words = sentence.split('\n')

            # go over the words (not including the start and end words)
            for i in range(window_size, len(words) - window_size):
                # for each word split into word and label
                word, ner = words[i].split('\t')

                # insert to the dataset a tuple of label and 5 words when the label is of the middle word
                dataset.append((ner, [word.split('\t')[0] for word in words[i - window_size: i + window_size + 1]]))

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


def parse_POS(file_path, window_size, pretrained=False):
    # initialize dictionaries
    word2index = {'<S>': 0, '<E>': 1, '<U>': 2}
    index2word = {0: '<S>', 1: '<E>', 2: '<U>'}
    label2index = {'<START>': 0, '<END>': 1, '<UNSEEN>': 2}
    index2label = {0: '<START>', 1: '<END>', 2: '<UNSEEN>'}

    with open(file_path, 'r', encoding='utf-8') as f:
        word_index = 2
        label_index = 2
        dataset = []

        # split into sentences (separated by blank rows)
        sentences = f.read().split('\n\n')

        for sentence in sentences:
            if sentence == '' or sentence == '\n':
                continue
            # add special words of start and end of sentence with special labels (<S> = START, <E> = END)
            sentence = '<S> START\n<S> START\n' + sentence + '\n<E> END\n<E> END'
            words = sentence.split('\n')

            # go over the words (not including the start and end words)
            for i in range(window_size, len(words) - window_size):
                # for each word split into word and label
                word, pos = words[i].split(' ')

                # insert to the dataset a tuple of label and 5 words when the label is of the middle word
                dataset.append((pos, [word.split(' ')[0] for word in words[i - window_size: i + window_size + 1]]))

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


def convert_dataset_to_index(dataset, word2index, label2index):
    for i in range(len(dataset)):
        # get current sample
        pos, words = dataset[i]
        # go over the words in the window
        for j in range(len(words)):
            # convert word to index. if the word was not seen - convert to unseen letter
            dataset[i][1][j] = word2index.get(words[j], word2index['<U>'])
        # change the tag to index
        dataset[i] = list(dataset[i])
        dataset[i][0] = label2index.get(pos, label2index['<UNSEEN>'])
        dataset[i] = tuple(dataset[i])

    return dataset


def parse_test_file(file_path, window_size):

    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = []

        # split into sentences (separated by blank rows)
        sentences = f.read().split('\n\n')

        for sentence in sentences:
            if sentence == '' or sentence == '\n':
                continue
            # add special words of start and end of sentence with special labels (<S> = START, <E> = END)
            sentence = '<S>\n<S>\n' + sentence + '\n<E>\n<E>'
            words = sentence.split('\n')

            # go over the words (not including the start and end words)
            for i in range(window_size, len(words) - window_size):
                # insert to the dataset a tuple of label and 5 words when the label is of the middle word
                dataset.append(words[i - window_size: i + window_size + 1])

    return dataset


def save_model(model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history, path):
    torch.save(model, f'{path}/model.path')
    torch.save(train_loss_history, f'{path}/train_loss_history.path')
    torch.save(train_accuracy_history, f'{path}/train_accuracy_history.path')
    torch.save(dev_loss_history, f'{path}/dev_loss_history.path')
    torch.save(dev_accuracy_history, f'{path}/dev_accuracy_history.path')


def load_model(model_path, train_loss_history_path, train_accuracy_history_path, dev_loss_history_path, dev_accuracy_history_path):
    model = torch.load(model_path)
    train_loss_history = torch.load(train_loss_history_path)
    train_accuracy_history = torch.load(train_accuracy_history_path)
    dev_loss_history = torch.load(dev_loss_history_path)
    dev_accuracy_history = torch.load(dev_accuracy_history_path)
    return model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history