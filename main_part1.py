import torch
from torch.utils.data import DataLoader

import utils
import tagger1


if __name__ == '__main__':
    pos_train_set, word2index, index2word, label2index, index2label = utils.parse_POS('./pos/train', window_size=2)
    pos_train_set = utils.convert_dataset_to_index(pos_train_set, word2index, label2index)

    pos_dev_set, _, _, _, _ = utils.parse_POS('./pos/dev', window_size=2)
    pos_dev_set = utils.convert_dataset_to_index(pos_dev_set, word2index, label2index)

    # ner_train_set, word2index, index2word, label2index, index2label = utils.parse_NER('./ner/train', window_size=2)
    # ner_train_set = utils.convert_dataset_to_index(ner_train_set, word2index, label2index)
    #
    # ner_dev_set, _, _, _, _ = utils.parse_NER('./ner/dev', window_size=2)
    # ner_dev_set = utils.convert_dataset_to_index(ner_dev_set, word2index, label2index)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_pos = True

    # define model's parameters
    vocab_size = len(word2index.keys())
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
        n_epochs = 8
        batch_size_train = 32
        batch_size_dev = 128
        hidden_dim = 150

    print(f'Run config - is POS: {is_pos}, vocab size: {vocab_size}, embed size: {embed_size}, window size: {num_words},'
          f' hidden layer size: {hidden_dim}, labels size: {out_dim}, LR: {lr}, epochs: {n_epochs},'
          f' train batch size: {batch_size_train}, dev batch size: {batch_size_dev}')


    # define train dataloader
    # train_data = DataLoader(ner_train_set, batch_size=batch_size_train, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    train_data = DataLoader(pos_train_set, batch_size=batch_size_train, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    # define train dataloader
    # dev_data = DataLoader(ner_dev_set, batch_size=batch_size_dev, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    dev_data = DataLoader(pos_dev_set, batch_size=batch_size_dev, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)

    model = tagger1.Tagger1Model(vocab_size, embed_size, num_words, hidden_dim, out_dim)

    tagger1.train_model(train_data, dev_data, model, n_epochs, lr, device, index2word, index2label, is_pos)

    # path = './pos results part 1'

    # model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history = utils.load_model(
    #     f'{path}/model.path', f'{path}/train_loss_history.path', f'{path}/train_accuracy_history.path',
    #     f'{path}/dev_loss_history.path', f'{path}/dev_accuracy_history.path'
    # )




