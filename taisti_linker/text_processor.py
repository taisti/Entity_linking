from nltk.stem import PorterStemmer
from typing import Any, Set
import re
import spacy


class TextProcessor:
    """ A class providing text-realted utilities """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.ps = PorterStemmer()

    def normalize_text(self, text: str) -> str:
        """
            Normalize ontology labels and NER outputs to increase the chance of a match.

            Args:
                text (str): text to normalize
            Returns:
                str: ormalized text
        """
        stopwords = ['the', 'a', 'an', 'at',
                     'by', 'for', 'in', 'into', 'on', 'to']
        # Hackish, in foodon default entities are annotated with (whole)
        text = re.sub(r"\(whole\)", "", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        text = " ".join([t for t in text.split(" ") if t not in stopwords])
        text = " ".join([self.ps.stem(token.text) for token in self.nlp(text)])
        return text
