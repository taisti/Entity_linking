from dataclasses import dataclass
from enum import Enum
from typing import List
import json
import os
import re
from xml.dom.minidom import Entity

class EntityType(Enum):
    FOOD = 1
    UNIT = 2
    QUANTITY = 3
    PROCESS = 4
    COLOR = 5
    PHYSICAL_QUALITY = 6
    DIET = 7
    PART = 8
    PURPOSE = 9
    TASTE = 10
 

class AnnotationSource(Enum):
    BRAT = 1
    NER = 2


@dataclass
class Annotation:
    id: str
    file_id: int
    start: int
    end: int
    category: str
    text: str
    source: AnnotationSource


@dataclass
class AnnotatedDoc:
    id: int
    path: str
    text: str
    annotations: List[Annotation]


@dataclass
class LabelWithIRI:
    label: str
    iri: str
    normalized_label: str


def get_entity_type(category: str) -> EntityType:
    category = category.lower()
    if 'food' in category or category in ['possible_substite', 'example', 'trade_name', 'excluded', 'exclusive']:
        return EntityType.FOOD
    elif category == 'unit':
        return EntityType.UNIT
    elif category == 'quantity':
        return EntityType.QUANTITY
    elif category == 'process':
        return EntityType.PROCESS
    elif category == 'color':
        return EntityType.COLOR
    elif category == 'physical_quality':
        return EntityType.PHYSICAL_QUALITY
    elif category == 'diet':
        return EntityType.DIET
    elif category == 'part':
        return EntityType.PART
    elif category == 'purpose':
        return EntityType.PURPOSE
    elif category == 'taste':
        return EntityType.TASTE


def read_brat_annotation_files(files_base_path: str) -> list[AnnotatedDoc]:
    annotations = []
    for filename in os.listdir(files_base_path):
        f = os.path.join(files_base_path, filename)
        if os.path.isfile(f) and f.endswith("txt"):
            ann_path = f"{f[:-4]}.ann"
            id = get_file_id(f)
            with open(f) as brat_file:
                text = brat_file.read()
            brat_annotations = read_brat_annotation(ann_path)
            annotated_doc = AnnotatedDoc(
                id=id, path=f, text=text, annotations=brat_annotations
            )
            annotations.append(annotated_doc)
    return annotations


def read_ner_annotation_file(path: str) -> list[AnnotatedDoc]:
    annotations = []
    with open(path) as f:
        documents = json.load(f)
        for i, doc in enumerate(documents):
            ner_annotations = []
            for j, entity in enumerate(doc['entities_list']):
                ner_annotations.append(Annotation(
                    id=str(j), file_id=i, start=entity['start'],
                    end=entity['end'], category=entity['label'],
                    text=entity['text'], source=AnnotationSource.NER))

            annotations.append(AnnotatedDoc(
                id=i, path=path, text=doc['text'], annotations=ner_annotations
            ))
    return annotations


def read_brat_annotation(path: str) -> List[Annotation]:
    annotations = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("T"):
                # filter annotations other than tokens
                id, details, text = line.strip().split("\t")
                if ";" in details:
                    # skip discontinuous annotations
                    continue

                category, start, end = details.split()
                file_id = get_file_id(path)
                annotations.append(
                    Annotation(
                        id=id,
                        file_id=file_id,
                        start=int(start),
                        end=int(end),
                        category=category,
                        text=text,
                        source=AnnotationSource.BRAT
                    )
                )
    return annotations


def get_file_id(path: str) -> int:
    return int(re.split(r"[./]", path)[-2])
