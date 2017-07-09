"""Tests for Language module

References:
    https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest

How to use:
go to test directory and run pytest (if you want to capture the output "pytest -s")
"""

import pytest
import sys
sys.path.insert(0, '../')
from Language import LanguageUtils

@pytest.fixture
def some_data():
    input_lang, output_lang, pairs = LanguageUtils.prepare_data('question', 'answer', reverse=True,
                                                                file_path='../data/train.txt')
    return input_lang, output_lang, pairs


def test_normalize_string():
    resp = LanguageUtils.normalize_string('Hi how are you?')
    assert resp == 'hi how are you ?'


def test_raises_exception_normalize_string():
    with pytest.raises(TypeError):
        resp = LanguageUtils.normalize_string(666)

def test_raises_exception_file_not_found_read_train_file():
    with pytest.raises(FileNotFoundError):
        input_lang, output_lang, pairs = LanguageUtils.read_train_file('lang1', 'lang2', False, 'bad_path')

def test_raises_exception_bad_parameter_read_train_file():
    with pytest.raises(TypeError):
        input_lang, output_lang, pairs = LanguageUtils.read_train_file('lang1', 'lang2', 'False', 'bad_path')


def test_small_data(some_data):
    input_lang, output_lang, pairs = some_data
    print (input_lang.sentence_2_indexes('thai food today'))