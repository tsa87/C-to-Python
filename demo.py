import encoder_decoder as model
from config import config
import train_eval
import torch
import utils

load_model = 0
training = 1
file_name = "with_attention_plot.png"

source_vocab, target_vocab, train_pairs, eval_pairs = utils.prepare_data()

if load_model:
    print("loading model...")
    encoder = torch.load(config.ENCODER_PATH)
    decoder = torch.load(config.DECODER_PATH)
else:
    print("initalizing model...")
    encoder = model.EncoderRNN(source_vocab.n_words, config.HIDDEN_SIZE)
    decoder = model.AttnDecoderRNN(target_vocab.n_words, config.HIDDEN_SIZE)

if training:
    train_loss_history,train_acc_history, eval_loss_history, eval_acc_history \
        = train_eval.trainIters(encoder, decoder, source_vocab, target_vocab, train_pairs, eval_pairs)

    utils.show_and_save_plot(
        train_loss_history,train_acc_history, eval_loss_history, eval_acc_history, file_name)

sentence = "BOOL && BOOL && BOOL"
translation, attention = train_eval.evaluate(
    encoder, decoder, source_vocab, target_vocab, sentence, max_length=config.MAX_LENGTH)
translation = utils.print_from_list(translation)
print(translation)
