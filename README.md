# Medbot

## Requirements
- **Python**:
```
spacy      3.2.1
torch      1.11.0
nltk       3.5
PyQt5      5.15.6
```
- `punkt` dataset for `nltk` library:
```
    $ python
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
$ python medbot_cli.py
```
or
```
$ python medbot_gui.py
```
