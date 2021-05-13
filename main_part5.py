import sys
import os

import torch
from torch.utils.data import DataLoader
import numpy as np

import utils
from tagger4 import Tagger4Model, train_model, predict, convert_dataset_to_index

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def pos():
    is_pretrained = True
    pos_train_set, word2index, index2word, label2index, index2label = utils.parse_POS('./pos/train', window_size=2)

    max_word_size = 20
    if is_pretrained:
        word2index, index2word = utils.get_pretrained_words()

    char2index, index2char = utils.create_char_inx_dict()
    pos_train_set = convert_dataset_to_index(pos_train_set, word2index, label2index, char2index, max_word_size,
                                             pretrained=is_pretrained)

    pos_dev_set, _, _, _, _ = utils.parse_POS('./pos/dev', window_size=2)
    pos_dev_set = convert_dataset_to_index(pos_dev_set, word2index, label2index, char2index, max_word_size,
                                           pretrained=is_pretrained)

    pos_test_set = utils.parse_test_file('./pos/test', window_size=2)
    pos_test_set = convert_dataset_to_index(pos_test_set, word2index, label2index, char2index, max_word_size,
                                            pretrained=is_pretrained, is_test=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_pos = True

    # define model's parameters
    vocab_size = len(word2index.keys())
    embed_size = 50
    c_embed_size = 20
    num_words = 5
    out_dim = len(label2index.keys())

    lr = 1e-4
    n_epochs = 10
    batch_size_train = 32
    batch_size_dev = 32
    hidden_dim = 150

    print(
        f'Run config - is POS: {is_pos}, vocab size: {vocab_size}, embed size: {embed_size}, window size: {num_words},'
        f' hidden layer size: {hidden_dim}, labels size: {out_dim}, LR: {lr}, epochs: {n_epochs},'
        f' train batch size: {batch_size_train}, dev batch size: {batch_size_dev}')

    # define train dataloader
    train_data = DataLoader(pos_train_set, batch_size=batch_size_train, shuffle=True, drop_last=True, num_workers=0)
    #
    # # define dev dataloader
    dev_data = DataLoader(pos_dev_set, batch_size=batch_size_dev, shuffle=False, drop_last=True, num_workers=0)

    embeddings = None
    if is_pretrained:
        embeddings = np.loadtxt('pretrained vectors.txt')

    model = Tagger4Model(vocab_size, embed_size, c_embed_size, num_words, hidden_dim, out_dim, max_word_size,
                         len(char2index.keys()), is_pretrained=is_pretrained, embeddings=embeddings)

    train_model(train_data, dev_data, model, n_epochs, lr, device, index2word, word2index, index2label, index2char, is_pos)

    """
    path = './pos results part 5'

    model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history = utils.load_model(
        model, f'{path}/model.path', f'{path}/train_loss_history.path', f'{path}/train_accuracy_history.path',
        f'{path}/dev_loss_history.path', f'{path}/dev_accuracy_history.path'
    )
    """

    # utils.draw_graphs(train_loss_history, dev_loss_history, n_epochs, 'Loss History', 'Train Loss', 'Validation Loss')
    # utils.draw_graphs(train_accuracy_history, dev_accuracy_history, n_epochs, 'Accuracy History', 'Train Accuracy', 'Validation Accuracy')

    test_data = DataLoader(pos_test_set, batch_size=1, shuffle=False, num_workers=0)

    predictions = predict(test_data, model, device, index2label, index2char, index2word, c_embed_size)

    utils.export_predictions(predictions, 'test5.pos')


def ner():
    is_pretrained = True
    ner_train_set, word2index, index2word, label2index, index2label = utils.parse_NER('./ner/train', window_size=2)

    max_word_size = 20
    if is_pretrained:
        word2index, index2word = utils.get_pretrained_words()

    char2index, index2char = utils.create_char_inx_dict()
    ner_train_set = convert_dataset_to_index(ner_train_set, word2index, label2index, char2index, max_word_size,
                                             pretrained=is_pretrained)

    ner_dev_set, _, _, _, _ = utils.parse_NER('./ner/dev', window_size=2)
    ner_dev_set = convert_dataset_to_index(ner_dev_set, word2index, label2index, char2index, max_word_size, pretrained=is_pretrained)

    ner_test_set = utils.parse_test_file('./ner/test', window_size=2)
    ner_test_set = convert_dataset_to_index(ner_test_set, word2index, label2index, char2index, max_word_size,
                                            pretrained=is_pretrained, is_test=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_pos = False

    # define model's parameters
    vocab_size = len(word2index.keys())
    embed_size = 50
    c_embed_size = 20
    num_words = 5
    out_dim = len(label2index.keys())

    lr = 1e-4
    n_epochs = 15
    batch_size_train = 32
    batch_size_dev = 32
    hidden_dim = 150

    print(
        f'Run config - is POS: {is_pos}, vocab size: {vocab_size}, embed size: {embed_size}, window size: {num_words},'
        f' hidden layer size: {hidden_dim}, labels size: {out_dim}, LR: {lr}, epochs: {n_epochs},'
        f' train batch size: {batch_size_train}, dev batch size: {batch_size_dev}')

    # define train dataloader
    train_data = DataLoader(ner_train_set, batch_size=batch_size_train, shuffle=True, drop_last=True, num_workers=0)

    # define dev dataloader
    dev_data = DataLoader(ner_dev_set, batch_size=batch_size_dev, shuffle=False, drop_last=True, num_workers=0)

    embeddings = None
    if is_pretrained:
        embeddings = np.loadtxt('pretrained vectors.txt')

    model = Tagger4Model(vocab_size, embed_size, c_embed_size, num_words, hidden_dim, out_dim, max_word_size,
                         len(char2index.keys()),
                         is_pretrained=is_pretrained, embeddings=embeddings)

    train_model(train_data, dev_data, model, n_epochs, lr, device, index2word, word2index, index2label, index2char, is_pos)

    # path = '.'
    #
    # model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history = utils.load_model(
    #     model, f'{path}/model.path', f'{path}/train_loss_history.path', f'{path}/train_accuracy_history.path',
    #     f'{path}/dev_loss_history.path', f'{path}/dev_accuracy_history.path'
    # )

    # define test dataloader
    test_data = DataLoader(ner_test_set, batch_size=1, shuffle=False, num_workers=0)

    predictions = predict(test_data, model, device, index2label, index2char, index2word, c_embed_size)

    utils.export_predictions(predictions, 'test5.ner')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Error with argument! Must be <task type>', file=sys.stderr)
    else:
        task = sys.argv[1]

        if task == 'pos':
            pos()
        elif task == 'ner':
            ner()
        else:
            print('Error with argument! Please check again', file=sys.stderr)
