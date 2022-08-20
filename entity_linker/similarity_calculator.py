from enum import Enum
from typing import Callable
from nltk.util import everygrams
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn


class SimilarityType(Enum):
    JACCARD = 1
    ANYGRAM = 2


class SimilarityCalculator:
    def __init__(self, similarity_type: SimilarityType, normalizer: Callable = None):
        self.similarity_type = similarity_type
        self.normalizer = normalizer

    def jaccard(self, text_a, text_b):
        text_a = self.normalizer(text_a)
        text_b = self.normalizer(text_b)  # decorator??
        a = set(text_a.split())
        b = set(text_b.split())

        if len(a) == 0 or len(b) == 0:
            return 0.0
        else:
            return 1.0 * len(a.intersection(b)) / len(a.union(b))

    def anygrams(self, text_a, text_b):
        text_a = self.normalizer(text_a)
        text_b = self.normalizer(text_b)  # decorator??
        a = set(everygrams(text_a.split()))
        b = set(everygrams(text_b.split()))

        if len(a) == 0 or len(b) == 0:
            return 0.0
        else:
            return 1.0 * len(a.intersection(b)) / len(a.union(b))

    def wordnet(self, text_a, text_b):
        text_a = pos_tag(word_tokenize(text_a))
        text_b = pos_tag(word_tokenize(text_b))

        # Get the synsets for the tagged words
        synsets1 = [self._tagged_to_synset(
            *tagged_word) for tagged_word in text_a]
        synsets2 = [self._tagged_to_synset(
            *tagged_word) for tagged_word in text_b]

        # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]

        score, count = 0.0, 0

        # For each word in the first sentence
        for synset in synsets1:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([synset.path_similarity(ss) for ss in synsets2])

            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1

        # Average the values
        score /= count
        return score

    def _penn_to_wn(self, tag):
        """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
        if tag.startswith('N'):
            return 'n'

        if tag.startswith('V'):
            return 'v'

        if tag.startswith('J'):
            return 'a'

        if tag.startswith('R'):
            return 'r'

        return None

    def _tagged_to_synset(self, word, tag):
        wn_tag = self._penn_to_wn(tag)
        if wn_tag is None:
            return None

        try:
            return wn.synsets(word, wn_tag)[0]
        except Exception:
            return None
