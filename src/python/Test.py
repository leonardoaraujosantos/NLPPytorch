"""Test sequence to sequence with Attention model

How to use:
python Test.py 'Hi do you know where I can find Indian food?'
"""

import fire
from Language import LanguageUtils
from Language import LangDef

# Pytorch includes
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# Encoder and Decoder includes
from Seq2SeqAttention import EncoderGRU
from Seq2SeqAttention import AttentionDecoderGRU

# Check if cuda is available and populate flag accordingly.
use_cuda = torch.cuda.is_available()

def evaluate(encoder, decoder, sentence, max_length=LangDef.max_words):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[LangDef.StartToken]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == LangDef.EndToken:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def test(enc_file='encoder.pkl', dec_file='decoder.pkl', hidd_size=16, in_sequence=''):


    # Load models parameters and other information
    info_encoder = torch.load(enc_file)
    print("Loading model (Encoder): %s" % enc_file)
    info_decoder = torch.load(dec_file)
    print("Loading model (Decoder): %s" % dec_file)

    # Initialize Encoder and Decoder with loaded information
    encoder = EncoderGRU(info_encoder['in_size'], hidden_size=info_encoder['hidd_size'])
    decoder = AttentionDecoderGRU(hidden_size=info_decoder['hidd_size'], output_size=info_decoder['out_size'], n_layers=1, dropout_p=0.1)
    encoder.load_state_dict(info_encoder['state_dict'])
    decoder.load_state_dict(info_decoder['state_dict'])

    # Push encoder and decoder to the GPU
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # Evaluate some sequence of text and print output
    #output_words, attentions = evaluate(encoder, decoder, in_sequence)
    #print('input =', in_sequence)
    #print('output =', ' '.join(output_words))

if __name__ == '__main__':
  # Only expose the test function to the command line
  fire.Fire(test)