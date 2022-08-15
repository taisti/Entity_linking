# import owlready2
import prodigy
import spacy
import csv
import os

from pathlib import Path
from prodigy.models.ner import EntityRecognizer


def load_food_entities():
    entities_loc = Path("food_product_entities.csv")

    names = dict()
    descriptions = dict()

    with entities_loc.open("r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")

        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]

            names[qid] = name
            descriptions[qid] = desc

    return names, descriptions


nlp = spacy.load("en_core_web_lg")
name_dict, desc_dict = load_food_entities()
kb = spacy.kb.KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

import re
import itertools
from itertools import chain, combinations
import warnings
warnings. filterwarnings('ignore')
stopwords = nlp.Defaults.stop_words
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r + 1) for r in range(len(s)))

def generate_combinations(iterable):
    return list(powerset(iterable))

def generate_permutations(iterable):
    return list(itertools.permutations(iterable))

def generate_candidates(name, kb, qid):
    all_tokens = list(set(name.split(' ')))
    candidates = []
    if len(all_tokens) > 7:
        # kb.add_alias(alias=name, entities=[qid], probabilities=[1.0])
        candidates = [name]
    else:
        for combination in generate_combinations(all_tokens):
            for permutation in generate_permutations(combination):
                allowed_alias = " ".join(permutation).strip()
                if len(allowed_alias) > 0:
                    candidates.append(allowed_alias)
    return candidates
                    #probability = 1.0 * len(list(permutation)) / len(all_tokens)
                    # kb.add_alias(alias=allowed_alias, entities=[qid], probabilities=[probability])

def normalize_name(name):
    name = re.sub(r'[^a-zA-Z]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.lower()
    name = ' '.join([token for token in name.split(' ') if token not in stopwords])
    stemmed_name = " ".join([ps.stem(token.text) for token in nlp(name)])
    return stemmed_name

for qid, desc in desc_dict.items():
    desc_doc = nlp(desc)
    desc_enc = desc_doc.vector
    kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)


idx = 0
aliases_to_qids = dict()
for qid, name in name_dict.items():
    print(f"processing: {idx} {name}")
    idx += 1
    aliases = generate_candidates(normalize_name(name), kb, qid)
    for alias in aliases:
        if not alias in aliases_to_qids:
            aliases_to_qids[alias] = []
        aliases_to_qids[alias].append(qid)

for alias, qids in aliases_to_qids.items():
    kb.add_alias(alias=alias, entities=qids, probabilities=[1.0/len(qids) for i in qids])

# save KnowlegeBase
output_dir = Path.cwd() / "output_food"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

kb.to_disk(output_dir / "my_kb")
nlp.to_disk(output_dir / "my_nlp")
