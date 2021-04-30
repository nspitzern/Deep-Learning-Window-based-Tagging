import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import utils


def cosine_calc(target_vec, vec):
    cosine = np.dot(target_vec, vec) / (LA.norm(vec) * LA.norm(target_vec))

    return cosine


def most_similar(target_word, k):
    words2vecs, words2inx = utils.create_word_vec_dict()
    sim = []
    target_vec = words2vecs[target_word]
    for vec in words2vecs.values():
        sim.append(cosine_calc(target_vec, vec))

    sim = np.array(sim)
    top_sim = [list(words2inx.keys())[list(words2inx.values()).index(i)] for i in sim.argsort()[-(k+1):-1][::-1]]
    dists = sim[sim.argsort()[-(k+1):-1][::-1]]

    return top_sim, dists


if __name__ == '__main__':
    k = 5
    target_words = ['dog', 'england', 'john', 'explode', 'office']

    for word in target_words:
        top_sim, dists = most_similar(word, k)
        print('Top-', str(k), ' similar words to ', word, ':')
        for close_word, dist in zip(top_sim, dists):
            print(close_word, ' : ', dist)
        print()
