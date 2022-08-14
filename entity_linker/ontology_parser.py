import owlready2
from commons import LabelWithIRI
from text_processor import TextProcessor
from typing import Dict


class OntologyParser:
    def __init__(self, ontology_path):
        self.ontology = owlready2.get_ontology(ontology_path).load()
        self.food_product_obj = owlready2.IRIS[
            "http://purl.obolibrary.org/obo/FOODON_00001002"
        ]

    def get_possible_labels(self, obj):
        prop_names = [
            "http://www.geneontology.org/formats/oboInOwl#hasSynonym",
            "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
            "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
            "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
            "http://purl.obolibrary.org/obo/IAO_0000118",  # alternative term
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
            for label in [self._get_label(c)]:
                label_to_iri[label] = c.iri
        return label_to_iri

    def get_IRI_labels_data(
        self, normalizator: TextProcessor
    ) -> Dict[str, LabelWithIRI]:
        result: Dict[str, LabelWithIRI] = dict()

        for c in self.food_product_obj.descendants():
            for label in [self._get_label(c)]:
                normalized_label = normalizator.normalize_text(label)
                if normalized_label in result:
                    print(f"WARNING: {normalized_label} already in mapping")
                result[normalized_label] = \
                    LabelWithIRI(label, c.iri, normalized_label)
        return result

    def _get_label(self, obj):
        """Return best label for given element.
        Args:
            obj (Any): object to get label from
        Returns:
            str: label of a given object
        """
        if hasattr(obj, "prefLabel") and obj.prefLabel.first() is not None:
            label = obj.prefLabel.first()
        elif hasattr(obj, "label") and obj.label.first() is not None:
            label = obj.label.first()
        elif hasattr(obj, "name"):
            label = obj.name
        else:
            label = "<UNKWNOWN>"
        return str(label)
