import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from vocab import Vocabulary
from config import config
import random
import torch
import time
import math
import numpy as np

def prepare_data():
    lines = open(config.TXT_DATA).read().strip().split('\n')
    pairs = [[snippet for snippet in line.split("$")] for line in lines]

    source_vocab = Vocabulary(config.SOURCE)
    target_vocab = Vocabulary(config.TARGET)

    for pair in pairs:
        source_vocab.add_sentence(pair[0])
        target_vocab.add_sentence(pair[1])

    random.shuffle(pairs)

    eval_pairs = pairs[:int(len(pairs)*config.EVAL_PERCENTAGE)]
    train_pairs = pairs[int(len(pairs)*config.EVAL_PERCENTAGE):]

    return source_vocab, target_vocab, train_pairs, eval_pairs

def tensor_from_sentence(vocab, sentence):
    indices = [vocab.word2index[word] for word in sentence.split()]
    indices.append(config.EOS_TOKEN)
    return torch.tensor(indices, dtype=torch.long).view(-1, 1)

def tensor_from_pair(pair, source_vocab, target_vocab):
    source_tensor = tensor_from_sentence(source_vocab, pair[0])
    target_tensor = tensor_from_sentence(target_vocab, pair[1])
    return (source_tensor,target_tensor)

def pair_from_tensor(vocab, tensor):
    sentence_vector = tensor.view(-1).tolist()
    sentence = [vocab.index2word[index] for index in sentence_vector]
    sentence = print_from_list(sentence)
    return sentence

def print_from_list(array):
    output = ""
    for i in range(len(array)-1):
        output = output + array[i]
    return output

def compute_accuracy(output, target):
    correct_counter = 0
    for i in range(len(target)):
        if i < len(output):
            if target[i] == output[i]:
                correct_counter += 1

    return correct_counter/len(target)

def show_and_save_plot(train_loss_history, train_acc_history, eval_loss_history, eval_acc_history, file_name):
    plt.subplot(2,1,1)
    plt.title('Loss')
    t = range(len(train_loss_history))
    plt.plot(t, train_loss_history,'r')
    plt.plot(t, eval_loss_history,'b')

    plt.subplot(2,1,2)
    plt.title('Accuracy')
    plt.ylim(0, 1)
    t = range(len(train_acc_history))
    plt.plot(t, train_acc_history,'r')
    plt.plot(t, eval_acc_history,'b')

    plt.savefig(file_name)
    print("Plot saved.")
    plt.show()
