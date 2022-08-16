import owlready2
from commons import EntityType, LabelWithIRI
from text_processor import TextProcessor
from typing import Dict


class OntologyParser:
    def __init__(self, ontology_path):
        self.ontology = owlready2.get_ontology(ontology_path).load()
        self.type_to_root_entity = self._get_root_nodes_for_categories()

    def _get_root_nodes_for_categories(self):
        return {
            EntityType.FOOD: owlready2.IRIS[
               "http://purl.obolibrary.org/obo/FOODON_00001002"
            ],
            EntityType.PROCESS: owlready2.IRIS[
                "http://purl.obolibrary.org/obo/FOODON_03530206"
            ]
        }

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

    def get_IRI_labels_data(
        self, normalizer: TextProcessor, category: EntityType
    ) -> Dict[str, LabelWithIRI]:
        result: Dict[str, LabelWithIRI] = dict()

        root = self.type_to_root_entity[category]

        for c in root.descendants():
            for label in [self._get_label(c)]:
                normalized_label = normalizer.normalize_text(label)
                if normalized_label in result:
                    print(f"WARNING: {normalized_label} already in mapping")
                result[normalized_label] = \
                    LabelWithIRI(label, c.iri, normalized_label)
        return result

    def get_IRI_labels_data_per_category(
        self, normalizer: TextProcessor
    ) -> Dict[EntityType, Dict[str, LabelWithIRI]]:
        result = dict()

        for entity_type in EntityType:
            if entity_type in self.type_to_root_entity:
                result[entity_type] = self.get_IRI_labels_data(normalizer, entity_type)
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
