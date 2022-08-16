from typing import Optional
from commons import (EntityType, LabelWithIRI, get_entity_type,
                     read_brat_annotation_files,
                     read_ner_annotation_file)
from ontology_parser import OntologyParser
from text_processor import TextProcessor

import argparse
import csv
import pickle
import os


class EntityLinker:
    def __init__(
        self,
        ontology_path: str,
        annotated_examples_base_path: str,
        ner_output_path: str,
        min_acceptable_similarity: float = 0.5,
        ignore_not_linkable: bool = False
    ):
        self.ontology_path = ontology_path
        self.annotated_examples_base_path = annotated_examples_base_path
        self.ner_output_path = ner_output_path
        self.min_acceptable_similarity = min_acceptable_similarity
        self.ignore_not_linkable = ignore_not_linkable
        self.ontology_parser = OntologyParser(ontology_path)
        self.text_processor = TextProcessor()

        if len(ner_output_path) > 0:
            self.annotated_docs = read_ner_annotation_file(ner_output_path)
        else:
            self.annotated_docs = \
                read_brat_annotation_files(annotated_examples_base_path)

        self.normalized_label_mapping = \
            self.generate_label_mapping(self.text_processor)

    def generate_label_mapping(self, text_processor):
        cache_path = './foodon_cache.pkl'

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            normalized_label_mapping = \
                self.ontology_parser.get_IRI_labels_data_per_category(
                    normalizer=text_processor
                )
            with open(cache_path, 'wb') as f:
                pickle.dump(normalized_label_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
            return normalized_label_mapping

    def link_all(self, output_path: str):
        print(f"INFO: Writing output to: {output_path}")
        f = open(output_path, "w")
        writer = csv.writer(f)

        for id, doc in enumerate(self.annotated_docs):
            for annotation_id, annotation in enumerate(doc.annotations):
                print(f"Processing step {id}/{annotation_id}")
                entity_text = annotation.text
                entity_type = get_entity_type(annotation.category)
                normalized_entity_text = \
                    self.text_processor.normalize_text(entity_text)

                linked_item: Optional[LabelWithIRI] = None
                if entity_type in self.normalized_label_mapping and normalized_entity_text in self.normalized_label_mapping[entity_type]:
                    linked_item = \
                        self.normalized_label_mapping[entity_type][normalized_entity_text]
                    print(f"Direct match of {entity_text} to {linked_item}")
                else:
                    linked_item = self.link(
                        normalized_entity_text, entity_type
                    )

                annotation_data = [
                    annotation.file_id,
                    annotation.id,
                    annotation.category,
                    annotation.start,
                    annotation.end,
                    annotation.text,
                    annotation.source
                ]
                if linked_item:
                    writer.writerow(
                        annotation_data + [linked_item.iri, linked_item.label]
                    )
                elif not self.ignore_not_linkable:
                    writer.writerow(annotation_data + ["NONE", "NONE"])

    def link(
        self, text: str, entity_type: EntityType
    ) -> Optional[LabelWithIRI]:
        best_item = None
        max_similarity = -1.0

        if entity_type not in self.normalized_label_mapping:
            return best_item

        category_label_mapping = self.normalized_label_mapping[entity_type]

        # if len(category_label_mapping) == 0:
        #    print(f"WARNING: Empty list of potential labels to link. Possibly no mapping for {entity_type} is provided in OntologyParser._get_root_nodes_for_categories()")

        for _, item in category_label_mapping.items():
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
                print(f"Found new best: {normalized_label} with similarity {current_label_similarity}")

        return best_item


def main(ontology_path, annotations_path, output_file_path, ner_output, ignore_not_linkable):
    el = EntityLinker(ontology_path, annotations_path, ner_output, ignore_not_linkable=ignore_not_linkable)
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

    parser.add_argument('-ner', '--ner_output',
                        help='Use NER output stored in a given file to process',
                        type=str,
                        default='')

    parser.add_argument('-out', '--output_file_path',
                        help='Path to generated output file',
                        type=str,
                        default='./report.csv')
    parser.add_argument('-ignore', '--ignore_not_linkable',
                        help='Do not serialize entities that cannot be linked',
                        type=bool,
                        default=False)

    args = parser.parse_args()
    main(args.ontology_path, args.annotations_path, args.output_file_path,
         args.ner_output, args.ignore_not_linkable)
