import matplotlib.pyplot as plt
import encoder_decoder as model
from config import config
import train_eval
import torch
import utils

load_model = 0
training = 1

source_vocab, target_vocab, train_pairs, eval_pairs = utils.prepare_data()

if load_model:
    print("loading model...")
    encoder = torch.load(config.ENCODER_PATH)
    decoder = torch.load(config.DECODER_PATH)
else:
    print("initalizing model...")
    encoder = model.EncoderRNN(source_vocab.n_words, config.HIDDEN_SIZE)
    decoder = model.DecoderRNN(target_vocab.n_words, config.HIDDEN_SIZE)

if training:
    train_loss_history, train_acc_history, eval_loss_history, eval_acc_history = train_eval.trainIters(encoder, decoder, source_vocab, target_vocab, train_pairs, eval_pairs)

    plt.subplot(2,1,1)
    t = range(len(train_loss_history))
    plt.plot(t, train_loss_history)
    plt.plot(t, eval_loss_history)

    plt.subplot(2,1,2)
    t = range(len(train_acc_history))
    plt.plot(t, train_acc_history)
    plt.plot(t, eval_acc_history)

    plt.show()

sentence = "BOOL && BOOL && BOOL"
translation = train_eval.evaluate(encoder, decoder, source_vocab, target_vocab, sentence, max_length=config.MAX_LENGTH)
translation = utils.print_from_list(translation)

print(translation)
