from typing import Dict, Optional
from commons import LabelWithIRI, read_annotation_files
from ontology_parser import OntologyParser
from text_processor import TextProcessor

import argparse
import csv


class EntityLinker:
    def __init__(
        self,
        ontology_path: str,
        annotated_examples_base_path: str,
        min_acceptable_similarity: float = 0.5,
    ):
        self.ontology_path = ontology_path
        self.annotated_examples_base_path = annotated_examples_base_path
        self.min_acceptable_similarity = min_acceptable_similarity
        self.ontology_parser = OntologyParser(ontology_path)
        self.text_processor = TextProcessor()

        self.annotated_docs = \
            read_annotation_files(annotated_examples_base_path)
        self.normalized_label_mapping = \
            self.ontology_parser.get_IRI_labels_data(
                normalizator=self.text_processor
            )

    def link_all(self, output_path: str):
        print(f"INFO: Writing output to: {output_path}")
        f = open(output_path, "w")
        writer = csv.writer(f)

        for id, doc in enumerate(self.annotated_docs):
            for annotation_id, annotation in enumerate(doc.annotations):
                print(f"Processing step {id}/{annotation_id}")
                entity_text = annotation.text
                normalized_entity_text = \
                    self.text_processor.normalize_text(entity_text)

                linked_item: Optional[LabelWithIRI] = None
                if normalized_entity_text in self.normalized_label_mapping:
                    linked_item = \
                        self.normalized_label_mapping[normalized_entity_text]
                else:
                    linked_item = self.link(
                        normalized_entity_text, self.normalized_label_mapping
                    )

                annotation_data = [
                    annotation.file_id,
                    annotation.id,
                    annotation.category,
                    annotation.start,
                    annotation.end,
                    annotation.text,
                ]
                if linked_item:
                    writer.writerow(
                        annotation_data + [linked_item.iri, linked_item.label]
                    )
                else:
                    writer.writerow(annotation_data + ["NONE", "NONE"])

    def link(
        self, text: str, normalized_label_mapping: Dict[str, LabelWithIRI]
    ) -> Optional[LabelWithIRI]:
        best_item = None
        max_similarity = -1.0

        for _, item in normalized_label_mapping.items():
            label = item.label
            normalized_label = item.normalized_label
            iri = item.iri
            current_label_similarity = self.text_processor.set_similarity(
                set(text.split()), set(normalized_label.split())
            )

            if (
                current_label_similarity > self.min_acceptable_similarity
                and current_label_similarity >= max_similarity
            ):
                max_similarity = current_label_similarity
                best_item = LabelWithIRI(label, iri, normalized_label)

        return best_item


def main(ontology_path, annotations_path, output_file_path):
    el = EntityLinker(ontology_path, annotations_path)
    el.link_all(output_file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-op', '--ontology_path',
                        help='Path to ontology that we want to link to',
                        type=str,
                        default='../foodon.owl')
    parser.add_argument('-ap', '--annotations_path',
                        help='Path to BRAT annotations folder',
                        type=str,
                        default='../data/')
    parser.add_argument('-out', '--output_file_path',
                    help='Path to generated output file',
                    type=str,
                    default='./report.csv')
    
    args = parser.parse_args()
    main(args.ontology_path, args.annotations_path, args.output_file_path)
