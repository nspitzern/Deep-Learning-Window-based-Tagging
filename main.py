import utils


if __name__ == '__main__':
    # pos_train_set, words2index, index2words, _, _ = utils.parse_POS('./pos/train', window_size=2)
    ner_train_set, words2index, index2words, _, _ = utils.parse_NER('./ner/train', window_size=2)
    # test_set = utils.parse_test_file('./pos/test', window_size=2)
    pass