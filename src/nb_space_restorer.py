"""main.py

Defines the NBSpaceRestorer class"""

import operator
import pickle
from collections import Counter
from functools import reduce
from memo.memo import memo
from math import log10

import nltk

ERROR_INIT_OVERSPECIFIED = """
Only one of train_texts and load_path should be specified.
Do you want to train a new model or load from a pickle file?"""


# ====================
class NBSpaceRestorer():

    # ====================
    def __init__(self, 
                 train_texts: list = None,
                 save_path: str = None,
                 load_path: str = None,
                 L: int = 20,
                 lambda_: float = 10.0,
                 ignore_case: bool = True):
        """Initialize a new class instance"""

        self.unigram_freqs = Counter()
        self.bigram_freqs = Counter()
        self.L = L
        self.lambda_ = lambda_

        if train_texts is not None and load_path is not None:
            raise ValueError(ERROR_INIT_OVERSPECIFIED)
        
        # Train from texts
        if train_texts:
            for text in train_texts:
                if ignore_case:
                    text = text.lower()
                words = text.split()
                self.unigram_freqs.update(words)
                bigrams = [f'{first_word}_{second_word}' 
                        for first_word, second_word in list(nltk.bigrams(words))]
                self.bigram_freqs.update(bigrams)
            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(
                        {'unigram_freqs': self.unigram_freqs, 'bigram_freqs': self.bigram_freqs},
                        f
                    )
        # Load unigram and bigram frequences from a file
        if load_path:
            freqs = pickle.load(load_path)
            self.unigram_freqs = freqs['unigram_freqs']
            self.bigram_freqs = freqs['bigram_freqs']

    # ====================
    def get_pdists(self):
        """Get unigram and bigram probability distributions from unigram
        and bigram frequencies"""

        # Get total numbers of unigrams and bigrams
        self.N = sum(self.unigram_freqs.values())
        self.N2 =  sum(self.bigram_freqs.values())
        # Define probability distributions for unigrams and bigrams
        self.Pdist = {word: freq / self.N
                      for word, freq in self.unigram_freqs.items()}
        self.P2dist = {bigram: freq / self.N2
                       for bigram, freq in self.bigram_freqs.items()}

    # ====================
    def splits(self, text) -> list:
        """Split text into a list of candidate (word, remainder) pairs."""
        return [(text[:i+1], text[i+1:]) for i in range(min(len(text), self.L))]

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
            # The number 10 is arbitrary and can be adjusted
            return self.lambda_/(self.N * 10 ** len(word))

    # ====================
    def cPw(self, word: str, prev: str) -> float:
        """Get conditional probability of a word given the previous word"""

        try:
            return self.P2dist[prev + '_' + word] / float(self.Pw(prev))
        except KeyError:
            return self.Pw(word)

    # ====================
    @memo
    def segment_recur(self, text_: str, prev='<S>') -> list:
        """Segment a sequence of characters through recursion"""

        if not text_:
            return 0.0, []
        candidates = [self.combine(log10(self.cPw(first, prev)),
                                   first,
                                   self.segment_recur(rem, first))
                      for first, rem in self.splits(text_)]
        return max(candidates)

    # ====================
    def combine(self, Pfirst: float, first: str, rem_: list):
        """Combine a single word and its probability with a list of remaining
        words and its probability to output a (float, list) tuple."""

        Prem, rem = rem_
        return Pfirst + Prem, [first] + rem

    # ====================
    def segment(self, text: str, show_chunks=False) -> list:
        """Segment a string of characters into words

        For strings over a certain length, break them into chunks for
        segmentation and then put the words back together to avoid
        recursion limit errors.
        """

        chunk_len_chars = 80   # Low enough to avoid recurision errors
        all_words = []
        prefix = ''
        chunk_counter = 1
        # Iterate over chunks of the input string
        for offset in range(0, len(text), chunk_len_chars):
            # Prefix with the last five 'words' from the previous segmentation
            text_to_segment = prefix + text[offset:offset + chunk_len_chars]
            chunk_segmented = self.segment_recur(text_to_segment)[1].copy()
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
        all_words.extend(self.segment_recur(prefix)[1])

        return all_words

    # ====================
    def restore_spaces(self, text: str) -> str:
        """Restore spaces to a sequence of characters"""

        words = self.segment(text)
        joined = ' '.join(words).strip()
        return joined

    # ====================
    def all_words_seen(self, text: str) -> bool:
        """Check whether all words in a string have been seen by the model
        during training"""

        words = text.split()
        if all([word in self.Pdist for word in words]):
            return True
        else:
            return False
