from config import config
from torch import optim
import torch.nn as nn
import torch
import utils
import time
import random

def trainIters(encoder, decoder, source_vocab, target_vocab, train_pairs, eval_pairs):

    train_loss_total = 0
    train_acc_total = 0
    eval_loss_total = 0
    eval_acc_total = 0
    train_acc_history = []
    train_loss_history = []
    eval_acc_history = []
    eval_loss_history = []

    start_time = time.time()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=config.LEARNING_RATE)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.NLLLoss()
    training_pairs = [utils.tensor_from_pair(random.choice(train_pairs),source_vocab,target_vocab)
                      for i in range(config.N_INTERATIONS)]

    eval_pairs = [utils.tensor_from_pair(random.choice(eval_pairs),source_vocab,target_vocab)
                      for i in range(config.N_INTERATIONS)]

    for iter in range(config.N_INTERATIONS):
        training_pair = training_pairs[iter]

        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        train_loss, train_acc = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        train_loss_total += train_loss
        train_acc_total += train_acc

        eval_pair = eval_pairs[iter]
        input_tensor = eval_pair[0]
        target_tensor = eval_pair[1]

        eval_loss, eval_acc = eval(input_tensor, target_tensor, encoder, decoder, criterion)
        eval_loss_total += eval_loss
        eval_acc_total += eval_acc


        if iter % config.AVG_EVERY == 0:
            train_loss_avg = train_loss_total/config.AVG_EVERY
            train_acc_avg = train_acc_total/config.AVG_EVERY
            train_loss_history.append(train_loss_avg)
            train_acc_history.append(train_acc_avg)

            eval_loss_avg = eval_loss_total/config.AVG_EVERY
            eval_acc_avg = eval_acc_total/config.AVG_EVERY
            eval_loss_history.append(eval_loss_avg)
            eval_acc_history.append(eval_acc_avg)

            train_loss_total = 0
            train_acc_total = 0
            eval_loss_total = 0
            eval_acc_total = 0

            percentage = round(iter/config.N_INTERATIONS, 4) * 100
            time_left = (time.time()-start_time)/(percentage)*(100-percentage) if iter != 0 else 0
            print("t_loss:{:.2f},t_acc:{:.2f},e_loss:{:.2f},e_acc:{:.2f},#: {}/{}, est_time: {:.2f}s"\
                   .format(train_loss_avg,train_acc_avg,eval_loss_avg,eval_acc_avg,iter,config.N_INTERATIONS,time_left))

    return train_loss_history, train_acc_history, eval_loss_history, eval_acc_history

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder.train()
    decoder.train()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(config.MAX_LENGTH, encoder.hidden_size)
    encoder_hidden = encoder.initHidden()

    loss = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[config.SOS_TOKEN]])
    decoder_hidden = encoder_hidden

    output = []

    if random.random() < config.TEACHER_FORCE_RATIO:
        # Teacher forcing: Feed the target as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            output.append(decoder_output.topk(1)[1].item())

            loss = loss + criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1)
            output.append(topi.item())

            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[i])

            if decoder_input.item() == config.EOS_TOKEN:
                break

    acc = utils.compute_accuracy(output, target_tensor.view(-1).tolist())

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    torch.save(encoder,config.ENCODER_PATH)
    torch.save(decoder,config.DECODER_PATH)

    return loss.item()/target_length, acc

def eval(input_tensor, target_tensor, encoder, decoder, criterion):

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(config.MAX_LENGTH, encoder.hidden_size)
    encoder_hidden = encoder.initHidden()

    eval_loss = 0
    eval_acc = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[config.SOS_TOKEN]])
    decoder_hidden = encoder_hidden

    output = []

    for i in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,encoder_outputs)

        topv, topi = decoder_output.topk(1)
        output.append(topi.item())

        decoder_input = topi.squeeze().detach()  # detach from history as input
        eval_loss += criterion(decoder_output, target_tensor[i])

        if decoder_input.item() == config.EOS_TOKEN:
            break

        eval_acc = utils.compute_accuracy(output, target_tensor.view(-1).tolist())

    return eval_loss.item()/target_length, eval_acc


def evaluate(encoder, decoder, source_vocab, target_vocab, sentence, max_length=config.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = utils.tensor_from_sentence(source_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i],encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]

        decoder_input = torch.tensor([[config.SOS_TOKEN]])

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[i] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == config.EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(target_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attention[:i+1]
