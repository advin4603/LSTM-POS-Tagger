# LSTM-POS-Tagger
A Part-of-Speech Tagger using LSTMs

## Requirements:
- python >= 3.10
- pytorch

Additional requirements in [requirements.txt](requirements.txt). Install them using:
```shell
pip install -r requirements.txt
```
## Demo Usage using an English Dataset

### Data
 Download Data from [here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4923).
- In the `ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-{train,dev,test}.conllu` delete the sentences containing the SYM tag.
