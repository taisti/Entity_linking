from enum import Enum
from typing import Any, Callable, List, Optional, Set
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

    def calculate(self, repr_a: Any, repr_b: Any) -> float:
        """
            Based on a similiraty measure id (either `j`, `e` or `w` for Jaccard, Everygram, Wordnet)
            calculate appropriate similarity measure.

            Args:
                repr_a (Any): left-hand-side similarity argument
                repr_b (Any): right-hand-side similarity argument
            Returns:
                float: similarity score
        """

        if self.similarity_type == SimilarityType.JACCARD:
            return self._jaccard(repr_a, repr_b)
        elif self.similarity_type == SimilarityType.EVERYGRAM:
            return self._everygrams(repr_a, repr_b)
        elif self.similarity_type == SimilarityType.WORDNET:
            return self._wordnet(repr_a, repr_b)

    def preprocess(self, text: str, normalize: bool = False) -> Any:
        """
            Based on a similiraty measure id (either `j`, `e` or `w` for Jaccard, Everygram, Wordnet)
            prepare the required representation for similarity calculation.

            Args:
                text (str): text to preprocess
                normalize (bool): whether to apply preprocesisng (False by default)
            Returns:
                Any: preprocessed representation
        """

        if self.similarity_type == SimilarityType.JACCARD:
            return self._jaccard_preprocess(text, normalize)
        elif self.similarity_type == SimilarityType.EVERYGRAM:
            return self._everygrams_preprocess(text, normalize)
        elif self.similarity_type == SimilarityType.WORDNET:
            return self._wordnet_preprocess(text, normalize)

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

    def _jaccard_preprocess(self, text: str, normalize: bool = False) -> Any:
        if normalize:
            text = self.normalizer(text)
        return set(text.split())

    def _everygrams_preprocess(self, text: str, normalize: bool = False) -> Any:
        if normalize:
            text = self.normalizer(text)
        return set(everygrams(text.split()))

    def _wordnet_preprocess(self, text: str, normalize: bool = False) -> Any:
        text = pos_tag(word_tokenize(text))
        synsets = [self._tagged_to_synset(
            *tagged_word) for tagged_word in text]
        return [ss for ss in synsets if ss]

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        """
            Jaccard based similarity between two texts represented as sets of unigrams.

            Args:
                a (Set[str]): first argument
                b (Set[str]): second argument
            Returns:
                float: Jaccard similarity score over sets
        """

        if len(a) == 0 or len(b) == 0:
            return 0.0
        else:
            return 1.0 * len(a.intersection(b)) / len(a.union(b))

    def _everygrams(self, a: Set[str], b: Set[str]) -> float:
        """
            Jaccard based similarity between two texts represented as everygrams.

            Args:
                a (Set[str]): first argument
                b (Set[str]): second argument
            Returns:
                float: Jaccard similarity score between everygrams.
        """

        if len(a) == 0 or len(b) == 0:
            return 0.0
        else:
            return 1.0 * len(a.intersection(b)) / len(a.union(b))

    def _wordnet(self, synsets1: List[Any], synsets2: List[Any]) -> float:
        """
            Wordnet based similarity between two texts.

            Args:
                synsets1 (List[Any]): left-hand-side argument
                synsets2 (List[Any]): right-hand-side argument
            Returns:
                float: Wordnet (path_similarity) similarity score between texts.
        """
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
