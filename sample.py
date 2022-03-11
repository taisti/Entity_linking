import csv
from pathlib import Path

import prodigy
import spacy
from prodigy.models.ner import EntityRecognizer


@prodigy.recipe(
    "entity_linker.manual",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .txt file", "positional", None, Path),
    nlp_dir=("Path to the NLP model with a pretrained NER component", "positional", None, Path),
    kb_loc=("Path to the KB", "positional", None, Path),
    entity_loc=("Path to the file with additional information about he entities", "positional", None, Path),
)
def entity_linker_manual(dataset, source, nlp_dir, kb_loc, entity_loc):
    nlp = spacy.load(nlp_dir)
    kb = spacy.kb.KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk(kb_loc)
    model = EntityRecognizer(nlp)

    id_dict = {}
    with entity_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            id_dict[row[0]] = (row[1], row[2])

    stream = prodigy.components.loaders.TXT(source)
    stream = [prodigy.util.set_hashes(eg) for eg in stream]
    stream = (eg for score, eg in model(stream))

    stream = _add_option(stream, kb, id_dict)
    stream = prodigy.components.filters.filter_duplicates(stream, by_input=True, by_task=False)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "choice",
        "config": {"choice_auto_accept": True},
    }


def _add_option(stream, kb, id_dict):
    for task in stream:
        text = task["text"]

        for span in task["spans"]:
            start_char = int(span["start"])
            end_char = int(span["end"])
            mention = text[start_char:end_char]

            candidates = kb.get_alias_candidates(mention)
            if candidates:
                options = [{"id": c.entity_, "html": _print_url(c.entity_, id_dict)} for c in candidates]
                options = sorted(options, key=lambda r: r["id"])
                options.append({"id": "NIL_otherLink", "text": "Link not in options"})
                options.append({"id": "NIL_ambiguous", "text": "Need more context"})
                task["options"] = options
                yield task


def _print_url(entity_id, id_dict):
    url_prefix = "https://www.wikidata.org/wiki"
    name, descr = id_dict.get(entity_id)
    option = "<a href='" + url_prefix + entity_id + "'>" + entity_id + "</a>: " + name + ':' + descr
    return option
