import torch
from torch.utils.data import DataLoader
import os
import sys

import utils
from tagger2 import Tagger2Model, train_model, predict, convert_dataset_to_index

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def pos():
    pos_train_set, word2index, index2word, label2index, index2label = utils.parse_POS('./pos/train', window_size=2)
    word2vec, _ = utils.create_word_vec_dict()
    pos_train_set = convert_dataset_to_index(pos_train_set, word2vec, label2index)
    
    pos_dev_set, _, _, _, _ = utils.parse_POS('./pos/dev', window_size=2)
    pos_dev_set = convert_dataset_to_index(pos_dev_set, word2vec, label2index)
    
    pos_test_set = utils.parse_test_file('./pos/test', window_size=2)
    pos_test_set = convert_dataset_to_index(pos_test_set, word2vec, label2index, is_test=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_pos = True

    # define model's parameters
    vocab_size = len(word2index.keys()) - 1
    embed_size = 50
    num_words = 5
    out_dim = len(label2index.keys())

    if is_pos:
        lr = 1e-4
        n_epochs = 10
        batch_size_train = 32
        batch_size_dev = 32
        hidden_dim = 150
    else:
        lr = 1e-3
        n_epochs = 6
        batch_size_train = 32
        batch_size_dev = 128
        hidden_dim = 100

    print(
        f'Run config - is POS: {is_pos}, vocab size: {vocab_size}, embed size: {embed_size}, window size: {num_words},'
        f' hidden layer size: {hidden_dim}, labels size: {out_dim}, LR: {lr}, epochs: {n_epochs},'
        f' train batch size: {batch_size_train}, dev batch size: {batch_size_dev}')

    # define train dataloader
    train_data = DataLoader(pos_train_set, batch_size=batch_size_train, shuffle=True, drop_last=True, num_workers=4)

    # define dev dataloader
    dev_data = DataLoader(pos_dev_set, batch_size=batch_size_dev, shuffle=False, drop_last=True, num_workers=4)

    model = Tagger2Model(vocab_size, embed_size, num_words, hidden_dim, out_dim)

    train_model(train_data, dev_data, model, n_epochs, lr, device, index2word, word2index, index2label, is_pos)

    # path = '.'
    #
    # model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history = utils.load_model(
    #     model, f'{path}/model.path', f'{path}/train_loss_history.path', f'{path}/train_accuracy_history.path',
    #     f'{path}/dev_loss_history.path', f'{path}/dev_accuracy_history.path'
    # )

    test_data = DataLoader(pos_test_set, batch_size=1, shuffle=False, num_workers=4)

    predictions = predict(test_data, model, device, index2label)

    utils.export_predictions(predictions, 'test3.pos')


def ner():
    ner_train_set, word2index, index2word, label2index, index2label = utils.parse_NER('./ner/train', window_size=2)
    word2vec, _ = utils.create_word_vec_dict()
    ner_train_set = utils.convert_dataset_to_index(ner_train_set, word2vec, label2index)

    ner_dev_set, _, _, _, _ = utils.parse_NER('./ner/dev', window_size=2)
    ner_dev_set = utils.convert_dataset_to_index(ner_dev_set, word2vec, label2index)

    ner_test_set = utils.parse_test_file('./ner/test', window_size=2)
    ner_test_set = convert_dataset_to_index(ner_test_set, word2vec, label2index, is_test=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_pos = False

    # define model's parameters
    vocab_size = len(word2index.keys()) - 1
    embed_size = 50
    num_words = 5
    out_dim = len(label2index.keys())

    if is_pos:
        lr = 1e-3
        n_epochs = 7
        batch_size_train = 32
        batch_size_dev = 32
        hidden_dim = 150
    else:
        lr = 1e-3
        n_epochs = 6
        batch_size_train = 32
        batch_size_dev = 128
        hidden_dim = 100

    print(
        f'Run config - is POS: {is_pos}, vocab size: {vocab_size}, embed size: {embed_size}, window size: {num_words},'
        f' hidden layer size: {hidden_dim}, labels size: {out_dim}, LR: {lr}, epochs: {n_epochs},'
        f' train batch size: {batch_size_train}, dev batch size: {batch_size_dev}')

    # define train dataloader
    train_data = DataLoader(ner_train_set, batch_size=batch_size_train, shuffle=True, drop_last=True, num_workers=4)

    # define dev dataloader
    dev_data = DataLoader(ner_dev_set, batch_size=batch_size_dev, shuffle=False, drop_last=True, num_workers=4)

    model = Tagger2Model(vocab_size, embed_size, num_words, hidden_dim, out_dim)

    train_model(train_data, dev_data, model, n_epochs, lr, device, index2word, word2index, index2label, is_pos)

    # path = '.'
    #
    # model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history = utils.load_model(
    #     model, f'{path}/model.path', f'{path}/train_loss_history.path', f'{path}/train_accuracy_history.path',
    #     f'{path}/dev_loss_history.path', f'{path}/dev_accuracy_history.path'
    # )

    test_data = DataLoader(ner_test_set, batch_size=1, shuffle=False, num_workers=4)

    predictions = predict(test_data, model, device, index2label)

    utils.export_predictions(predictions, 'test3.ner')

    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Error with argument! Must be <task type> <is pretrained>', file=sys.stderr)
    else:
        task, is_pretrained = sys.argv[1:]

        if task == 'pos':
            pos()
        elif task == 'ner':
            ner()
        else:
            print('Error with argument! Please check again', file=sys.stderr)
