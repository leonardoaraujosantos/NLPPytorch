# Introduction
Doing some NLP with Pytorch

### Install python requirements
```bash
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
pip install torchvision

git clone https://github.com/pytorch/text.git
cd text
python setup.py install
```

### Run Vanilla C++11 matcher
Just build (qmake) project at __src/cpp14/vanilla_matcher__
``` bash
qmake vanilla_matcher.pro
make
./vanilla_matcher

I would like some Thai food
thai
-----------END QUESTION-----------
Where can I find good sushi
sushi
-----------END QUESTION-----------
Find me a place that does tapas
NONE
-----------END QUESTION-----------
Which restaurants do East Asian food
east
asian
-----------END QUESTION-----------
Which restaurants do West Indian food
west
indian
-----------END QUESTION-----------
What is the weather like today
NONE
-----------END QUESTION-----------
===============================================================================
test cases: 1 | 1 passed
assertions: - none -

```
### Train model
From folder __src/python/__ just run:
```bash
python Train.py --print_every=100
```

### Run tests (Python sequence to sequence with attention)
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
