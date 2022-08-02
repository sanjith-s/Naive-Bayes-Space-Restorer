# Naive-Bayes-Space-Restorer

A Python class to allow enable convenient training of a Naive Bayes-based statistical model for restoration of spaces to unsegmented streams of input characters.

E.g.
`thisisasentence -> this is a sentence`

The model is based on the description and Python code in Norvig (2009), and the chunking methods for handling long strings is borrowed from Jenks (2018).

This implementation allows for easy restoration of spaces to entire datasets of documents with a progress bar, and for tuning of hyperparameters _L_ (maximum word length) and λ (smoothing parameter) for model optimization.

## Getting started

### 1. Clone the repository

Recommended method for Google Colab notebooks:

```python
import sys
# Delete naive-bayes-space-restorer folder to ensures that any changes to the repo are reflected
!rm -rf 'naive-bayes-space-restorer'
# Clone naive-bayes-space-restorer repo
!git clone https://github.com/ljdyer/naive-bayes-space-restorer.git
# Add naive-bayes-space-restorer/src to PYTHONPATH
sys.path.append('naive-bayes-space-restorer/src')
```

### 2. Install requirements

IF working in Google Colab, the only requirement is `python-memo`. All other requirements are installed by default.

```python
!pip install python-memo
```

If working in a virtual environment, run the following in the src directory:

```python
pip install -r requirements.txt
```

### 3. Import NBSpaceRestorer class

```python
from nb_space_restorer import NBSpaceRestorer
```

## How to use

### Initialize a class instance





## References

G. Jenks, ”python-wordsegment,” July, 2018. [Online]. Available:
https://github.com/grantjenks/python-wordsegment. [Accessed May
2, 2022].

P. Norvig, “Natural language corpus data,” in Beautiful Data, T.
Segaran and J. Hammerbacher, Eds. Sebastopol: O’Reilly, 2009, pp.
219-242.