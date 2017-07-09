"""Provide vectors from words on a text

Helper class used to to tokenize and convert it's tokens(words) into
vector representations (glove, one-hot, etc...)
"""

import fire
import unicodedata
import re


# Define start-of-sequence and end-of-sequence
class LangDef:
    Start = 0
    End = 1
    max_words=10


class LanguageUtils:
    def __init__(self, name=''):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {LangDef.Start: "SOS", LangDef.End: "EOS"}
        self.n_words = 2  # Count SOS and EOS


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
        # Make lower
        s = LanguageUtils.unicode_2_ascii(in_string.lower().strip())
        # Trim punctiation
        s = re.sub(r"([.!?])", r" \1", in_string)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", in_string)
        return s

    @staticmethod
    def read_train_file(in_string, out_string, reverse=False, file_path='data/train.txt'):
        """ Open train file and convert it's words to vectors
        The filename is expected to have 2 sentences per line separated by TAB
        """
        print("Reading lines...")

        # Read the file and and convert to a list on each line
        lines = open(file_path, encoding='utf-8').read().strip().split('\n')

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
    def prepare_data(lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = LanguageUtils.read_train_file(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = LanguageUtils.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs


# Using fire to help test from console without need to create command line parsers
# python Language.py normalize-string --in_string='Hi, how are you? Do you know some places to eat?'
# python Language.py unicode_2_ascii 'Hi, how are you? Do you know some places to eat?'
# python Language.py read_train_file input output
# python Language.py prepare_data input output
if __name__ == '__main__':
  fire.Fire(LanguageUtils)