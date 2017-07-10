# Introduction
Doing some NLP with Pytorch

### Run tests
From folder __src/python/test__ execute "pytest -s"
```bash
platform linux -- Python 3.5.2, pytest-3.0.5, py-1.4.32, pluggy-0.4.0
rootdir: /home/laraujo/work/NLPPytorch/src/python/test, inifile: 
collected 14 items 

test_Language.py ....Counting words...
Counted words:
answer 32
question 10
[6, 7, 31]
.
test_model_eval.py .Loading model (Encoder): ../encoder.pkl
.Loading model (Encoder): ../encoder.pkl
Loading model (Decoder): ../decoder.pkl
Trained words: 10
.Loading model (Encoder): ../encoder.pkl
Loading model (Decoder): ../decoder.pkl
Trained words: 10
input = which restaurants do east asian food .
output = east asian . <EOS>
.Loading model (Encoder): ../encoder.pkl
Loading model (Decoder): ../decoder.pkl
Trained words: 10
input = I would like some thai food.
output = thai . <EOS>
.Loading model (Encoder): ../encoder.pkl
Loading model (Decoder): ../decoder.pkl
Trained words: 10
input = Find me a place that does tapas.
output = none . <EOS>
.Loading model (Encoder): ../encoder.pkl
Loading model (Decoder): ../decoder.pkl
Trained words: 10
input = Which restaurants do West Indian food.
output = west indian indian . <EOS>
.Loading model (Encoder): ../encoder.pkl
Loading model (Decoder): ../decoder.pkl
Trained words: 10
input = What is the weather like today.
output = none . <EOS>
.Loading model (Encoder): ../encoder.pkl
Loading model (Decoder): ../decoder.pkl
Trained words: 10
word: nao not on dictionary, replace with EOS
word: conheco not on dictionary, replace with EOS
word: esta not on dictionary, replace with EOS
word: mensagem not on dictionary, replace with EOS
input = Nao conheco esta mensagem .
output = none . <EOS>
.

```
