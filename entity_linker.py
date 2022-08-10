import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import owlready2
import csv
from nltk.stem import PorterStemmer
import spacy


@dataclass
class BratAnnotation:
    id: str
    file_id: int
    start: int
    end: int
    category: str
    text: str


@dataclass
class AnnotatedDoc:
    id: int
    path: str
    text: str
    annotations: List[BratAnnotation]


@dataclass
class LabelWithIRI:
    label: str
    iri: str


class OntologyParser:
    def __init__(self, ontology_path):
        self.ontology = owlready2.get_ontology(ontology_path).load()
        self.food_product_obj = owlready2.IRIS["http://purl.obolibrary.org/obo/FOODON_00001002"]

    def get_possible_labels(self, obj):
        prop_names = [
            'http://www.geneontology.org/formats/oboInOwl#hasSynonym',
            'http://www.geneontology.org/formats/oboInOwl#hasExactSynonym',
            'http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym',
            'http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym',
            'http://purl.obolibrary.org/obo/IAO_0000118',  # alternative term
        ]
        synonyms = [self._get_label(obj)]
        for prop_name in prop_names:
            prop = owlready2.IRIS[prop_name]
            if prop in obj.get_properties(obj):
                synonyms += prop[obj]
        return list(set(synonyms))

    def get_label_to_iri_mapping(self):
        label_to_iri = dict()
        for c in self.food_product_obj.descendants():
            for label in self.get_possible_labels(c):
                label_to_iri[label] = c.iri
        return label_to_iri

    def _get_label(self, obj):
        """ Return best label for given element.
        Args:
            obj (Any): object to get label from
        Returns:
            str: label of a given object
        """
        if hasattr(obj, 'prefLabel') and obj.prefLabel.first() is not None:
            label = obj.prefLabel.first()
        elif hasattr(obj, 'label') and obj.label.first() is not None:
            label = obj.label.first()
        elif hasattr(obj, 'name'):
            label = obj.name
        else:
            label = "<UNKWNOWN>"
        return label


class TextComparator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.ps = PorterStemmer()

    def normalize_text(self, text: str) -> str:
        stopwords = self.nlp.Defaults.stop_words
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = ' '.join(
            [token for token in text.split(' ') if token not in stopwords])
        text = " ".join([self.ps.stem(token.text) for token in self.nlp(text)])
        return text

    def similarity(self, text_a: str, text_b: str) -> float:
        normalized_text_a = set(self.normalize_text(text_a).split())
        normalized_text_b = set(self.normalize_text(text_b).split())

        return 1.0 * len(normalized_text_a.intersection(normalized_text_b)) / len(normalized_text_a.union(normalized_text_b))

class EntityLinker:
    def __init__(self, ontology_path: str, annotated_examples_base_path: str,
                 min_acceptable_similarity: int = 0.5):
        self.ontology_path = ontology_path
        self.annotated_examples_base_path = annotated_examples_base_path
        self.min_acceptable_similarity = min_acceptable_similarity
        self.ontology_parser = OntologyParser(ontology_path)
        self.text_comparator = TextComparator()

        self.annotated_docs = self.read_annotation_files(annotated_examples_base_path)
        self.ontology_label_to_iri = self.ontology_parser.get_label_to_iri_mapping()

    def link_all(self, output_path: str):
        f = open(output_path, 'w')
        writer = csv.writer(f)

        for id, doc in enumerate(self.annotated_docs):
            for annotation_id, annotation in enumerate(doc.annotations):
                print(f"Processing step {id}/{annotation_id}")
                entity_text = annotation.text

                linked_item = self.link(entity_text, self.ontology_label_to_iri)
                annotation_data = [annotation.file_id, annotation.id, annotation.category, annotation.start, annotation.end, annotation.text]
                if linked_item:
                    writer.writerow(annotation_data + [linked_item.iri, linked_item.label])
                else:
                    writer.writerow(annotation_data + ['NONE', 'NONE'])

    def link(self, text: str, label_to_iri: Dict[str, str]) -> Optional[LabelWithIRI]:
        best_item = None
        max_similarity = -1.0

        for label, iri in label_to_iri.items():
            current_label_similarity = self.text_comparator.similarity(text, label)
            if (current_label_similarity > self.min_acceptable_similarity and
                    current_label_similarity > max_similarity):
                max_similarity = current_label_similarity
                best_item = LabelWithIRI(label, iri)

        return best_item


    @staticmethod
    def read_annotation_files(
            annotated_examples_base_path: str) -> list[AnnotatedDoc]:

        annotations = []
        for filename in os.listdir(annotated_examples_base_path):
            f = os.path.join(annotated_examples_base_path, filename)
            if os.path.isfile(f) and f.endswith('txt'):
                ann_path = f"{f[:-4]}.ann"
                id = EntityLinker.get_file_id(f)
                with open(f) as brat_file:
                    text = brat_file.read()
                brat_annotations = EntityLinker.read_brat_annotation(ann_path)
                annotated_doc = AnnotatedDoc(
                    id=id, path=f, text=text,
                    annotations=brat_annotations)
                print(annotated_doc)
                annotations.append(annotated_doc)
        return annotations

    @staticmethod
    def read_brat_annotation(path: str) -> List[Tuple[int, int, str]]:
        annotations = []

        with open(path, 'r') as f:
            for line in f:
                if line.startswith('T'):
                    # filter annotations other than tokens
                    id, details, text = line.strip().split('\t')
                    if ";" in details:
                        # skip discontinuous annotations
                        continue

                    category, start, end = details.split()
                    file_id = EntityLinker.get_file_id(path)
                    annotations.append(BratAnnotation(id=id,
                                                      file_id=file_id,
                                                      start=int(start),
                                                      end=int(end),
                                                      category=category,
                                                      text=text))
        return annotations

    @staticmethod
    def get_file_id(path: str) -> int:
        return int(re.split(r'[./]', path)[-2])


def main():
    el = EntityLinker('./foodon.owl', './data/')
    el.link_all('./report.csv')

if __name__ == "__main__":
    main()
