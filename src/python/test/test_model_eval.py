"""Tests for Model evaluation

How to use:
go to test directory and run pytest (if you want to capture the output "pytest -s")
"""

import pytest
import sys
sys.path.insert(0, '../')
import Test


def test_raises_exception_file_not_found_encoder():
    with pytest.raises(FileNotFoundError):
        Test.test(enc_file='badFile', dec_file='../decoder.pkl',
                  in_sentence='which restaurants do east asian food .')


def test_raises_exception_file_not_found_decoder():
    with pytest.raises(FileNotFoundError):
        Test.test(enc_file='../encoder.pkl', dec_file='badFile',
                  in_sentence='which restaurants do east asian food .')


def test_raises_exception_type_error():
    with pytest.raises(TypeError):
        Test.test(enc_file='../encoder.pkl', dec_file='../decoder.pkl',
                  in_sentence=0)


def test_model_message_1():
    # Fail if exception is raised
    Test.test(enc_file='../encoder.pkl', dec_file='../decoder.pkl',
                in_sentence='which restaurants do east asian food .')


def test_model_message_2():
    # Fail if exception is raised
    Test.test(enc_file='../encoder.pkl', dec_file='../decoder.pkl',
                in_sentence='I would like some thai food.')


def test_model_message_3():
    # Fail if exception is raised
    Test.test(enc_file='../encoder.pkl', dec_file='../decoder.pkl',
                in_sentence='Where can I find good sushi.')


def test_model_message_3():
    # Fail if exception is raised
    Test.test(enc_file='../encoder.pkl', dec_file='../decoder.pkl',
                in_sentence='Find me a place that does tapas.')


def test_model_message_4():
    # Fail if exception is raised
    Test.test(enc_file='../encoder.pkl', dec_file='../decoder.pkl',
                in_sentence='Which restaurants do West Indian food.')


def test_model_message_5():
    # Fail if exception is raised
    Test.test(enc_file='../encoder.pkl', dec_file='../decoder.pkl',
                in_sentence='What is the weather like today.')


def test_model_message_bad():
    # Fail if exception is raised
    Test.test(enc_file='../encoder.pkl', dec_file='../decoder.pkl',
                in_sentence='Nao conheco esta mensagem .')