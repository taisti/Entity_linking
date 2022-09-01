from turtle import st
import owlready2
from commons import EntityType, LabelWithIRI
from text_processor import TextProcessor
from typing import Any, Dict, List


class OntologyParser:
    """ A class for loading ontologies and managing label -> IRI maps """

    def __init__(self, ontology_path: str):
        self.ontology = owlready2.get_ontology(ontology_path).load()
        self.type_to_root_entity = self._get_root_nodes_for_categories()
        self.enabled_warnings = False

    def get_possible_labels(self, obj: Any) -> List[str]:
        """
            For a given ontology entity (owlready2 object) collect all possible labels (including synonyms)

            Args:
                obj (Any): object for which the synonyms should be collected
            Returns:
                List[str]: list of labels
        """
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
                synonyms += [str(s) for s in prop[obj]]
        return list(set(synonyms))

    def get_IRI_labels_data(
        self, normalizer: TextProcessor, category: EntityType
    ) -> Dict[str, LabelWithIRI]:
        """
            For a given category, generate a map considering all ontology entities matching this category.
            The map relates normalized entity labels with their IRIs

            Args:
                normalizer (TextProcessor): A normalizer that can transform labels into normalized forms.
                category (EntityType): A category for which the map should be constructed.
            Returns:
                Dict[str, LabelWithIRI]: A map of normalized labels to their IRIs
        """
        result: Dict[str, LabelWithIRI] = dict()

        root = self.type_to_root_entity[category]

        for c in root.descendants():
            for label in self.get_possible_labels(c):
                normalized_label = normalizer.normalize_text(label)
                if self.enabled_warnings and normalized_label in result:
                    print(f"WARNING: {normalized_label} already in mapping")
                result[normalized_label] = \
                    LabelWithIRI(label, c.iri, normalized_label)
        return result

    def get_IRI_labels_data_per_category(
        self, normalizer: TextProcessor
    ) -> Dict[EntityType, Dict[str, LabelWithIRI]]:
        """
            Calculate a map that for each NER/BRAT category (e.g., FOOD, COLOR, PROCESS)
            relates all allowed (normalized) entity labels to their IRIs.

            Args:
                normalizer (TextProcessor): A normalizer that can transform labels into normalized forms.
            Returns:
                Dict[EntityType, Dict[str, LabelWithIRI]]: For each category, a map of normalized labels to their IRIs
        """
        result = dict()

        for entity_type in EntityType:
            if entity_type in self.type_to_root_entity:
                result[entity_type] = self.get_IRI_labels_data(
                    normalizer, entity_type)
        return result

    def _get_label(self, obj: Any) -> str:
        """
            Return best label for given element.

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

    def _get_root_nodes_for_categories(self):
        """
            Each NER/BRAT annotation should be linked to a specific place in an ontology.
            For example, FOOD should be linked to FOOD only, not entities representing processes.
            For this reason, this function returns a map relating categories to roots of allowed taxonomies.

            Returns:
                Dict[EntityType, IRI]: Map relating NER/BRAT entity types to IRIs of entities the subclasses 
                                       of which entities are allowed to be linked to.
        """
        return {
            EntityType.FOOD: owlready2.IRIS[
                "http://purl.obolibrary.org/obo/FOODON_00001002"
            ],
            EntityType.PROCESS: owlready2.IRIS[
                "http://purl.obolibrary.org/obo/FOODON_03530206"
            ]
        }
