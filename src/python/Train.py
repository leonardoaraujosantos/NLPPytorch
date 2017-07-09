"""Train sequence to sequence with Attention model

On this example we will use the "Teacher Forcing" algorithm that uses the target output for the next input, instead
of the decoder output. Basically this make the training converge faster. (This kind of force a forget)

The only issue with "Teacher Forcing" is that you need to be sure that there is no dependecies between sentences.

References:
    * https://papers.nips.cc/paper/6099-professor-forcing-a-new-algorithm-for-training-recurrent-networks.pdf
    * http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf
    * https://www.quora.com/Should-you-use-teacher-forcing-in-LSTM-or-GRU-networks-is-the-forget-gate-sufficient
"""

import fire
import random
from Language import LangDef

# Pytorch includes
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# Check if cuda is available and populate flag accordingly.
use_cuda = torch.cuda.is_available()

teacher_forcing_ratio = 0.5


def train_step(in_var, label_var, encoder, decoder, enc_optim, dec_optimizer, criterion, max_length=LangDef.max_words):
    encoder_hidden = encoder.initHidden()

    # Zero gradients of encoder and decoder
    enc_optim.zero_grad()
    dec_optimizer.zero_grad()

    input_length = in_var.size()[0]
    target_length = label_var.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # Initialize loss
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            in_var[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[LangDef.StartToken]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    # Choose to use teacher forcing (Train will be called many times...)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for target_index in range(target_length):
            # Get decoder output, attention and hidden state
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)

            # Get loss
            loss += criterion(decoder_output[0], label_var[target_index])

            # Using targets as next input (Teacher forcing)
            decoder_input = label_var[target_index]

    else:
        # Without teacher forcing: Slower convergence (use decoder output as next input)
        for target_index in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)

            # Select decoder output
            topv, topi = decoder_output.data.topk(1)
            next_input = topi[0][0]

            # Put decoder output as next input
            decoder_input = Variable(torch.LongTensor([[next_input]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            # Get loss
            loss += criterion(decoder_output[0], label_var[target_index])

            # Stop if arrive on end of token
            if next_input == LangDef.EndToken:
                break

    loss.backward()

    enc_optim.step()
    dec_optimizer.step()

    return loss.data[0] / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Configure optimizer for decoder and encoder as Stochastic Gradient Descent
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]

    # Get negative log likelihood loss (Multinomial Cross entropy)
    criterion = nn.NLLLoss()

    # Iterate "iter" times
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train_step(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses