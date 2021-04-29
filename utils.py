import numpy as np
import torch


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

    for i in range(len(dataset)):
        pos, words = dataset[i]
        for j in range(len(words)):
            dataset[i][1][j] = word2index[words[j]]
        dataset[i] = list(dataset[i])
        dataset[i][0] = label2index[pos]
        dataset[i] = tuple(dataset[i])

    return dataset, word2index, index2word, label2index, index2label


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
