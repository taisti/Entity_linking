from nltk.stem import PorterStemmer
from typing import Any, Set
import re
import spacy


class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.ps = PorterStemmer()

    def normalize_text(self, text: str) -> str:
        stopwords = self.nlp.Defaults.stop_words
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        text = " ".join([t for t in text.split(" ") if t not in stopwords])
        text = " ".join([self.ps.stem(token.text) for token in self.nlp(text)])
        return text

    def similarity(self, text_a: str, text_b: str) -> float:
        print(f"Similarity between {text_a} {text_b}")
        normalized_text_a = set(self.normalize_text(text_a).split())
        normalized_text_b = set(self.normalize_text(text_b).split())

        return (
            1.0
            * len(normalized_text_a.intersection(normalized_text_b))
            / len(normalized_text_a.union(normalized_text_b))
        )

    def set_similarity(self, a: Set[Any], b: Set[Any]) -> float:
        if len(a) == 0 or len(b) == 0:
            return 0.0
        else:
            return 1.0 * len(a.intersection(b)) / len(a.union(b))
