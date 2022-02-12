# Medbot

## Requirements
- **Python**:
```
    python      3.9.2
    spacy       3.2.1
    torch       1.7.0a0
    nltk        3.5
```
- `punkt` dataset for `nltk` library:
```
    $ python3
    >>> import nltk
    >>> nltk.download('punkt')
```
- "en_core_web_trf" model for `spacy` library:
```
$ python -m spacy download en_core_web_trf
```


## Usage
- To train models:
```
$ python train.py
```
- To chat:
```
$ python medbot.py
```