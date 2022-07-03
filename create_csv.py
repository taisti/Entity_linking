from owlready2 import *
import csv

onto = get_ontology("foodon.owl").load()

f = open('food_product_entities_synonyms.csv', 'w')
writer = csv.writer(f)

def get_label(obj):
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

def get_possible_labels(obj):
    prop_names = [
        'http://www.geneontology.org/formats/oboInOwl#hasSynonym',
        'http://www.geneontology.org/formats/oboInOwl#hasExactSynonym',
        'http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym',
        'http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym',
        'http://purl.obolibrary.org/obo/IAO_0000118', # alternative term
    ]
    synonyms = [get_label(obj)]
    for prop_name in prop_names:
        prop = IRIS[prop_name]
        if prop in obj.get_properties(obj):
            synonyms += prop[obj]
    return list(set(synonyms))

food_product = IRIS["http://purl.obolibrary.org/obo/FOODON_00001002"]
textual_definition = IRIS["http://purl.obolibrary.org/obo/IAO_0000115"]

for c in food_product.descendants():
    for label in get_possible_labels(c):
        if textual_definition in c.get_properties(c):
            #print(textual_definition[c][-1])
            td = textual_definition[c][-1]
            #print(c, label, td)
            writer.writerow([c, label, td])
        else:
            #print(c, label, "There is no textual definition provided")
            writer.writerow([c, label, 'NONE'])
