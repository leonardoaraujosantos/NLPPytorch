"""Provide vectors from words on a text

Helper class used to to tokenize and convert it's tokens(words) into
vector representations (glove, one-hot, etc...)
"""

import fire
import unicodedata
import re

import torch
from torch.autograd import Variable

# Check if cuda is available and populate flag accordingly.
use_cuda = torch.cuda.is_available()

# Pytorch extension for NLP (Datasets, and some tools like word2vec and glove)
from torchtext.vocab import load_word_vectors


# Define start-of-sequence and end-of-sequence
class LangDef:
    StartToken = 0
    EndToken = 1
    max_words=10


class LanguageUtils:
    def __init__(self, name='', use_glove=False):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {LangDef.StartToken: "SOS", LangDef.EndToken: "EOS"}
        self.n_words = 2  # Count SOS and EOS

        if use_glove:
            # Get a dictionary of words (word to vec) with word vector size of 100 dimensions
            # It will download around 800Mb if necessary
            self.__wv_dict, self.__wv_arr, self.__wv_size = load_word_vectors('.', 'glove.6B', 100)
            print('Loaded', len(self.__wv_arr), 'words')

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence_2_indexes(self, sentence):
        list_index = []
        for word in sentence.split(' '):
            try:
                list_index.append(self.word2index[word])
            except:
                print('word:', word, 'not on dictionary, replace with EOS')
                list_index.append(LangDef.EndToken)

        return list_index
        #return [self.word2index[word] for word in sentence.split(' ')]

    def sentence_2_embeddings(self, sentence):
        list_index = []
        for word in sentence.split(' '):
            try:
                list_index.append(self.get_embeddings(word))
            except:
                print('word:', word, 'not on dictionary, replace with EOS')
                list_index.append(LangDef.EndToken)

        return list_index

    def get_embeddings(self, word):
        """ Glove word to vector
        """
        return self.__wv_arr[self.__wv_dict[LanguageUtils.normalize_string(word)]]

    def get_closest(self, d, n=10):
        """ Get closes n words
        Consider as a vec_2_word, that could return more than one element
        """
        all_dists = [(w, torch.dist(d, self.get_embeddings(w))) for w in self.__wv_dict]
        return sorted(all_dists, key=lambda t: t[1])[:n]

    @staticmethod
    def unicode_2_ascii(in_string):
        """ Convert unicode text to ascii

        Used to handle cases where input is not an ASCII string
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', in_string)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalize_string(in_string):
        """ Take out some punctiation and convert string to lowecase
        """
        # Check for valid input type
        if not isinstance(in_string, str):
            raise TypeError('Function expect a str type')

        # Make lower
        in_string = LanguageUtils.unicode_2_ascii(in_string.lower().strip())
        # Trim punctiation
        in_string = re.sub(r"([.!?])", r" \1", in_string)
        in_string = re.sub(r"[^a-zA-Z.!?]+", r" ", in_string)
        return in_string

    @staticmethod
    def read_train_file(in_string, out_string, reverse=False, file_path='data/train.txt'):
        """ Open train file and convert it's words to vectors
        The filename is expected to have 2 sentences per line separated by TAB
        """
        # Check for valid input type
        if not isinstance(reverse, bool):
            raise TypeError('Function parameter reverse expect a bool type')

        if not isinstance(in_string, str):
            raise TypeError('Function parameter in_string expect a str type')

        if not isinstance(out_string, str):
            raise TypeError('Function parameter out_string expect a str type')

        # Read the file and and convert to a list on each line
        try:
            lines = open(file_path, encoding='utf-8').read().strip().split('\n')
        except:
            raise FileNotFoundError('File %s not found', (file_path))

        # Split senteces separated by TAB on each line
        pairs = [[LanguageUtils.normalize_string(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = LanguageUtils(out_string)
            output_lang = LanguageUtils(in_string)
        else:
            input_lang = LanguageUtils(in_string)
            output_lang = LanguageUtils(out_string)

        return input_lang, output_lang, pairs

    @staticmethod
    def filter_pair(p):
        """ Check pair size
        Check if either pair is bigger than a maximum number of words
        """
        return len(p[0].split(' ')) < LangDef.max_words and len(p[1].split(' ')) < LangDef.max_words

    @staticmethod
    def filter_pairs(pairs):
        """ Check pairs size
        Filter out whole line of training if is bigger than max_words
        """
        return [pair for pair in pairs if LanguageUtils.filter_pair(pair)]

    @staticmethod
    def prepare_data(lang1, lang2, file_path='data/train.txt', reverse=False):
        input_lang, output_lang, pairs = LanguageUtils.read_train_file(lang1, lang2, reverse, file_path)
        pairs = LanguageUtils.filter_pairs(pairs)
        print("Counting words...")
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs

    @staticmethod
    def sentence_2_variable(lang, sentence):
        """ Convert sentence (streams of word vectors) to pytorch variable
        """
        indexes = lang.sentence_2_indexes(sentence)
        indexes.append(LangDef.EndToken)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        if use_cuda:
            return result.cuda()
        else:
            return result

    @staticmethod
    def pair_2_variable(in_lang, target_lang, pair):
        input_variable = LanguageUtils.sentence_2_variable(in_lang, pair[0])
        target_variable = LanguageUtils.sentence_2_variable(target_lang, pair[1])
        return (input_variable, target_variable)


# Using fire to help test from console without need to create command line parsers
# python Language.py normalize-string --in_string='Hi, how are you? Do you know some places to eat?'
# python Language.py unicode_2_ascii 'Hi, how are you? Do you know some places to eat?'
# python Language.py read_train_file input output
# python Language.py prepare_data input output
if __name__ == '__main__':
  fire.Fire(LanguageUtils)