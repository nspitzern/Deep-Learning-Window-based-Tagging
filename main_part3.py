import torch
from torch.utils.data import DataLoader
import os

import utils
import tagger2

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

if __name__ == '__main__':
    pos_train_set, word2index, index2word, label2index, index2label = utils.parse_POS('./pos/train', window_size=2)
    word2vec, _ = utils.create_word_vec_dict()
    pos_train_set = utils.convert_dataset_to_index(pos_train_set, word2vec, label2index)

    pos_dev_set, _, _, _, _ = utils.parse_POS('./pos/dev', window_size=2)
    pos_dev_set = utils.convert_dataset_to_index(pos_dev_set, word2vec, label2index)
    # ner_train_set, words2index, index2words, _, _ = utils.parse_NER('./ner/train', window_size=2)
    # test_set = utils.parse_test_file('./pos/test', window_size=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 1e-3
    n_epochs = 10
    batch_size = 32

    # define model's parameters
    vocab_size = len(word2index.keys())
    embed_size = 50
    num_words = 5
    hidden_dim = 128
    out_dim = len(label2index.keys())

    is_pos = True

    # define train dataloader
    train_data = DataLoader(pos_train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    # define train dataloader
    dev_data = DataLoader(pos_dev_set, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)

    model = tagger2.Tagger2Model(vocab_size, embed_size, num_words, hidden_dim, out_dim)

    tagger2.train_model(train_data, dev_data, model, n_epochs, lr, device, word2index, label2index, is_pos)

