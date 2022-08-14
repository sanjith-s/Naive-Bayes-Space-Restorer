"""main.py

Defines the NBSpaceRestorer class"""

import operator
import os
from collections import Counter
from functools import lru_cache, reduce
from math import log10

import nltk
import psutil

from nb_helper import (Str_or_List, get_tqdm, load_pickle,
                       mk_dir_if_does_not_exist, save_pickle)
from nb_space_restorer_grid_search import NBSpaceRestorerGridSearch

tqdm_ = get_tqdm()

FREQS_FNAME = 'FREQS.pickle'
GRID_SEARCH_PATH_NAME = 'grid_searches'

ERROR_MODEL_EXISTS = """\
There is already a NB Space Restorer at this path. \
Either choose a new path, or load the existing NB Space Restorer"""
MESSAGE_FINISHED_LOADING = "Finished loading model."
MESSAGE_TRAINING_COMPLETE = "Training complete."


# ====================
class NBSpaceRestorer():

    # ====================
    def __init__(self,
                 root_folder: str,
                 train_texts: list,
                 ignore_case: bool = True):
        """Initialize an instance of the NBSpaceRestorer class

        Required arguments:
        -------------------
        root_folder: str            The folder that will contain all instance
                                    assets.
                                    The folder should not already exist. It will
                                    be created automatically.

        train_texts: list           The list of 'gold standard' documents on
                                    which to train the model.

        Optional keyword arguments:
        ---------------------------
        ignore_case: bool           If True, case will be ignored during
            = True                  training (so that e.g. 'banana', 'Banana',
                                    and 'BANANA' are all counted as
                                    occurences of 'banana').
        """

        if os.path.exists(root_folder):
            raise ValueError(ERROR_MODEL_EXISTS)
        mk_dir_if_does_not_exist(root_folder)
        self.root_folder = root_folder
        self.unigram_freqs = Counter()
        self.bigram_freqs = Counter()
        for text in train_texts:
            if ignore_case:
                text = text.lower()
            words = text.split()
            self.unigram_freqs.update(words)
            bigrams = [
                f'{first_word}_{second_word}'
                for first_word, second_word in list(nltk.bigrams(words))
            ]
            self.bigram_freqs.update(bigrams)
        freqs = {
            'unigram_freqs': self.unigram_freqs,
            'bigram_freqs': self.bigram_freqs
        }
        save_pickle(freqs, self.freqs_path())
        print(MESSAGE_TRAINING_COMPLETE)
        self.get_pdists()

    # ====================
    @classmethod
    def load(cls, root_folder: str):
        """Load a previously saved instance of the NBSpaceRestorer class

        Required arguments:
        -------------------
        root_folder: str            The folder that contains all instance
                                    assets (i.e. the folder passed as
                                    root_folder when the instance was
                                    initialized).
        """

        self = cls.__new__(cls)
        self.root_folder = root_folder
        self.unigram_freqs, self.bigram_freqs = \
            self.get_freqs()
        print(MESSAGE_FINISHED_LOADING)
        self.get_pdists()
        return self

    # ====================
    def set_L(self, L: int):

        self.L = L

    # ====================
    def set_lambda_(self, lambda_: float):

        self.lambda_ = lambda_

    # ====================
    def freqs_path(self):

        return os.path.join(self.root_folder, FREQS_FNAME)

    # ====================
    def get_freqs(self):

        freqs = load_pickle(self.freqs_path())
        return freqs['unigram_freqs'], freqs['bigram_freqs']

    # ====================
    def get_pdists(self):
        """Get unigram and bigram probability distributions from unigram
        and bigram frequencies"""

        # Get total numbers of unigrams and bigrams
        self.N = sum(self.unigram_freqs.values())
        self.N2 = sum(self.bigram_freqs.values())
        # Define probability distributions for unigrams and bigrams
        self.Pdist = {word: freq / self.N
                      for word, freq in self.unigram_freqs.items()}
        self.P2dist = {bigram: freq / self.N2
                       for bigram, freq in self.bigram_freqs.items()}

    # ====================
    def splits(self, text) -> list:
        """Split text into a list of candidate (word, remainder) pairs."""
        return [
            (text[:i+1], text[i+1:]) for i in range(min(len(text), self.L))
        ]

    # ====================
    @staticmethod
    def product(lis_: list):
        """Product of a list of numbers"""

        return reduce(operator.mul, lis_, 1)

    # ====================
    def Pwords(self, words: list) -> float:
        """Get NB probability of a sequence of words"""

        return self.product(self.Pw(w) for w in words)

    # ====================
    def Pw(self, word: str) -> float:
        """Get NB probability of a single word"""

        if word in self.Pdist:
            return self.Pdist[word]
        else:
            # For unknown words, assign lower probabilities for longer words
            return self.lambda_/(self.N * 10 ** len(word))

    # ====================
    def cPw(self, word: str, prev: str) -> float:
        """Get conditional probability of a word given the previous word"""

        try:
            return self.P2dist[prev + '_' + word] / float(self.Pw(prev))
        except KeyError:
            return self.Pw(word)

    # ====================
    def combine(self, Pfirst: float, first: str, rem_: list):
        """Combine a single word and its probability with a list of remaining
        words and its probability to output a (float, list) tuple."""

        Prem, rem = rem_
        return Pfirst + Prem, [first] + rem

    # ====================
    @lru_cache(maxsize=None)
    def restore_chunk(self, text_: str, prev='<S>') -> list:
        """Restore spaces to a short string of input characters

        Will result in RecursionError if length of text_ is more than
        around 100."""

        if not text_:
            return 0.0, []
        candidates = [self.combine(log10(self.cPw(first, prev)),
                                   first,
                                   self.restore_chunk(rem, first))
                      for first, rem in self.splits(text_)]
        return max(candidates)

    # ====================
    def restore_doc(self, text: str, show_chunks=False) -> list:
        """Restore spaces to a string of input characters of arbitrary
        length.

        For strings over around 100 characters in length, break them
        into chunks for segmentation and then put the words back together
        to avoid recursion limit errors."""

        chunk_len_chars = 80   # Low enough to avoid recursion errors
        all_words = []
        prefix = ''
        chunk_counter = 1
        # Iterate over chunks of the input string
        for offset in range(0, len(text), chunk_len_chars):
            # Prefix with the last five 'words' from the previous segmentation
            text_to_segment = prefix + text[offset:offset + chunk_len_chars]
            chunk_segmented = self.restore_chunk(text_to_segment)[1].copy()
            # Words may have been cut off at the end, so put the last five
            # words back into the segmenter next time round and discard them
            # this time
            prefix = ''.join(chunk_segmented[-5:])
            all_words.extend(chunk_segmented[:-5])
            if show_chunks:
                print(f'Chunk {chunk_counter}')
                print(f'Text segmented: {text_to_segment}')
                print(f'Result of segmentation: {chunk_segmented}')
                print(f'Words added to list this time: {chunk_segmented[:-5]}')
                print(f'Prefix for next chunk: {prefix}')
                print(f'Words added to list so far: {all_words}')
                print('-' * 100)
                chunk_counter += 1
        # Add any text remaining in 'prefix'
        all_words.extend(self.restore_chunk(prefix)[1])
        joined = ' '.join(all_words).strip()
        return joined

    # ====================
    def restore(self,
                texts: Str_or_List,
                L: int = 20,
                lambda_: float = 10.0) -> str:
        """Restore spaces to either a single string, or a list of
        strings.

        If the input is a single string, the output will also be
        a single string.
        If the input is a list of strings, the output will be a
        list of the same length as the input.

        Required arguments:
        -------------------
        texts: Str_or_List          Either a single string of input
                                    characters, or a list of strings
                                    of input characters.
                                    Input strings should not contain
                                    spaces (e.g. 'thisisasentence')

        L: int = 20                 The maximum possible word length to
                                    consider during inference. Inference
                                    time increases with L as more probabilities
                                    need to be calculated.

        lambda_ = 10.0              The smoothing parameter to use during
                                    inference. Higher values of lambda_ cause
                                    higher probabilities to be assigned to
                                    words not learnt during training.
        """

        self.set_L(L)
        self.set_lambda_(lambda_)
        if isinstance(texts, str):
            return self.restore_doc(texts)
        if isinstance(texts, list):
            restored = []
            texts_ = tqdm_(texts)
            for text in texts_:
                restored_ = self.restore_doc(text)
                restored.append(restored_)
                texts_.set_postfix({
                    'ram_usage': f"{psutil.virtual_memory().percent}%"
                })
            return restored

    # === GRID SEARCH ===

    # ====================
    def add_grid_search(self,
                        grid_search_name: str,
                        L: list,
                        lambda_: list,
                        ref: list,
                        input: list):

        attrs = locals()
        del attrs['self']
        self.grid_search = NBSpaceRestorerGridSearch(self, **attrs)

    # ====================
    def load_grid_search(self,
                         grid_search_name: str):

        self.grid_search = \
            NBSpaceRestorerGridSearch.load(self, grid_search_name)

    # ====================
    def grid_search_path(self):

        return os.path.join(self.root_folder, GRID_SEARCH_PATH_NAME)
