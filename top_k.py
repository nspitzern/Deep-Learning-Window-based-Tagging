import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


def cosine_calc(target_vec, vec):
    cosine = np.dot(target_vec, vec) / (LA.norm(vec) * LA.norm(target_vec))

    return cosine


def most_similar(target_word, k):
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
    sim = []
    target_vec = words2vecs[target_word]
    for vec in words2vecs.values():
        sim.append(cosine_calc(target_vec, vec))

    sim = np.array(sim)
    top_sim = [list(words2inx.keys())[list(words2inx.values()).index(i)] for i in sim.argsort()[-6:-1][::-1]]

    return top_sim


if __name__ == '__main__':
    k = 5
    target_words = ['dog', 'england', 'john', 'explode', 'office']

    for word in target_words:
        top_sim = most_similar(word, k)
        print('Top-', str(k), ' similar words to ', word, ':')
        print(top_sim)
