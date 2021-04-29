import torch
from torch.utils.data import DataLoader

import utils
import tagger1


if __name__ == '__main__':
    pos_train_set, word2index, index2word, label2index, index2label = utils.parse_POS('./pos/train', window_size=2)
    pos_dev_set, _, _, _, _ = utils.parse_POS('./pos/dev', window_size=2)
    # ner_train_set, words2index, index2words, _, _ = utils.parse_NER('./ner/train', window_size=2)
    # test_set = utils.parse_test_file('./pos/test', window_size=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 1e-3
    n_epochs = 10
    batch_size = 2

    # define model's parameters
    vocab_size = len(word2index.keys())
    embed_size = 50
    num_words = 5
    hidden_dim = 128
    out_dim = len(label2index.keys())

    # define train dataloader
    train_data = DataLoader(pos_train_set, batch_size=batch_size, shuffle=True)

    # define train dataloader
    dev_data = DataLoader(pos_dev_set, batch_size=batch_size, shuffle=False)

    model = tagger1.Tagger1Model(batch_size, vocab_size, embed_size, num_words, hidden_dim, out_dim)

    tagger1.train_model(train_data, dev_data, model, n_epochs, lr, device, word2index, label2index)

