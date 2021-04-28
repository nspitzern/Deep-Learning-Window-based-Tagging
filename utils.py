import numpy as np
import torch

word2index = dict()
index2word = dict()


def parse_NER(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        i = 0

        lines = f.readlines()

        for line in lines:
            word, pos = line.split(' ')



def parse_POS(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        pass