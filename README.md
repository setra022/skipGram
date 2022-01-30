# skipGram

Train data used :

https://www.statmt.org/europarl/v7/fr-en.tgz

Usage :
```bash
python skipGram.py --text europarl-v7.fr-en.txt --model model.pickle
python skipGram.py --text train3.txt --model model.pickle
python skipGram.py --text SimLex-999\SimLex-999.txt --model model.pickle --test
```