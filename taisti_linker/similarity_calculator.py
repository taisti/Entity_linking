from enum import Enum
from typing import Callable, Optional
from nltk.util import everygrams
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn


class SimilarityType(Enum):
    JACCARD = 1
    EVERYGRAM = 2
    WORDNET = 3


class SimilarityCalculator:
    """ Similarity metrics container """

    def __init__(self, similarity_type: SimilarityType, normalizer: Callable = None):
        self.similarity_type = similarity_type
        self.normalizer = normalizer

    def calculate(self, text_a: str, text_b: str) -> float:
        """
            Based on a similiraty measure id (either `j`, `e` or `w` for Jaccard, Everygram, Wordnet)
            calculate appropriate similarity measure.

            Args:
                text_a (str): left-hand-side similarity argument
                text_b (str): right-hand-side similarity argument
            Returns:
                float: similarity score
        """

        if self.similarity_type == SimilarityType.JACCARD:
            return self._jaccard(text_a, text_b)
        elif self.similarity_type == SimilarityType.EVERYGRAM:
            return self._jaccard(text_a, text_b)
        elif self.similarity_type == SimilarityType.WORDNET:
            return self._jaccard(text_a, text_b)

    @staticmethod
    def similarity_id_to_type(similarity_measure_id: str = 'j') -> SimilarityType:
        """
            Transform textual representation of similarity id into appropriate type

            Args:
                similarity_measure_id (str): either j or J (for Jaccard), e or E (for Everygrams), w or W (for Wordnet)
                                             if unknown letter is provided, the jaccard similarity is used
            Returns:
                SimilarityType: Similarity type
        """
        similarity_measure_id = similarity_measure_id.lower()
        if similarity_measure_id == 'e':
            return SimilarityType.EVERYGRAM
        elif similarity_measure_id == 'w':
            return SimilarityType.WORDNET
        else:
            return SimilarityType.JACCARD

    def _jaccard(self, text_a: str, text_b: str) -> float:
        """
            Jaccard based similarity between two texts represented as sets of unigrams.

            Args:
                text_a (str): text to be transformed into the first set
                text_b (str): text to be transformed into the second set
            Returns:
                float: Jaccard similarity score over normalized unigrams
        """
        text_a = self.normalizer(text_a)
        text_b = self.normalizer(text_b)  # decorator??
        a = set(text_a.split())
        b = set(text_b.split())

        if len(a) == 0 or len(b) == 0:
            return 0.0
        else:
            return 1.0 * len(a.intersection(b)) / len(a.union(b))

    def _everygrams(self, text_a: str, text_b: str) -> float:
        """
            Jaccard based similarity between two texts represented as everygrams.

            Args:
                text_a (str): text to be transformed into the first set
                text_b (str): text to be transformed into the second set
            Returns:
                float: Jaccard similarity score between everygrams.
        """
        text_a = self.normalizer(text_a)
        text_b = self.normalizer(text_b)  # decorator??
        a = set(everygrams(text_a.split()))
        b = set(everygrams(text_b.split()))

        if len(a) == 0 or len(b) == 0:
            return 0.0
        else:
            return 1.0 * len(a.intersection(b)) / len(a.union(b))

    def _wordnet(self, text_a: str, text_b: str) -> float:
        """
            Wordnet based similarity between two texts.

            Args:
                text_a (str): left-hand-side argument
                text_b (str): right-hand-side argument
            Returns:
                float: Wordnet (path_similarity) similarity score between texts.
        """
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

    def _penn_to_wn(self, tag: str) -> Optional[str]:
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

    def _tagged_to_synset(self, word: str, tag: str):
        """ Extarct synsets for a given word and its POS-tag"""
        wn_tag = self._penn_to_wn(tag)
        if wn_tag is None:
            return None

        try:
            return wn.synsets(word, wn_tag)[0]
        except Exception:
            return None
