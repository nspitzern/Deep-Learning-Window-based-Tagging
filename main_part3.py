import torch
from torch.utils.data import DataLoader
import os

import utils
import tagger2

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

if __name__ == '__main__':
    # pos_train_set, word2index, index2word, label2index, index2label = utils.parse_POS('./pos/train', window_size=2)
    # word2vec, _ = utils.create_word_vec_dict()
    # pos_train_set = utils.convert_dataset_to_index(pos_train_set, word2vec, label2index, pretrained=True)
    #
    # pos_dev_set, _, _, _, _ = utils.parse_POS('./pos/dev', window_size=2)
    # pos_dev_set = utils.convert_dataset_to_index(pos_dev_set, word2vec, label2index, pretrained=True)
    #
    # pos_test_set = utils.parse_test_file('./pos/test', window_size=2)

    ner_train_set, words2index, index2words, label2index, index2label = utils.parse_NER('./ner/train', window_size=2)
    word2vec, _ = utils.create_word_vec_dict()
    ner_train_set = utils.convert_dataset_to_index(ner_train_set, word2vec, label2index, pretrained=True)

    ner_dev_set, _, _, _, _ = utils.parse_NER('./ner/dev', window_size=2)
    ner_dev_set = utils.convert_dataset_to_index(ner_dev_set, word2vec, label2index, pretrained=True)

    ner_test_set = utils.parse_test_file('./ner/test', window_size=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define model's parameters
    vocab_size = len(word2vec.keys())
    embed_size = 50
    num_words = 5
    out_dim = len(label2index.keys())

    is_pos = False

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
    # train_data = DataLoader(pos_train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    train_data = DataLoader(ner_train_set, batch_size=batch_size_train, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)


    # define train dataloader
    # dev_data = DataLoader(pos_dev_set, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    dev_data = DataLoader(ner_dev_set, batch_size=batch_size_dev, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)

    model = tagger2.Tagger2Model(vocab_size, embed_size, num_words, hidden_dim, out_dim)

    tagger2.train_model(train_data, dev_data, model, n_epochs, lr, device, index2label, is_pos)

    # path = './pos results part 3'

    # model, train_loss_history, train_accuracy_history, dev_loss_history, dev_accuracy_history = utils.load_model(
    #     model, f'{path}/model.path', f'{path}/train_loss_history.path', f'{path}/train_accuracy_history.path',
    #     f'{path}/dev_loss_history.path', f'{path}/dev_accuracy_history.path'
    # )

    for i in range(len(ner_test_set)):
        for j in range(len(ner_test_set[i])):
            ner_test_set[i][j] = word2vec.get(ner_test_set[i][j], word2vec['UUUNKKK'])

    # define test dataloader
    # test_data = DataLoader(ner_test_set, batch_size=1, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    test_data = DataLoader(ner_test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    predictions = tagger2.predict(test_data, model, device, index2label)

    utils.export_predictions(predictions, 'test3.ner')
