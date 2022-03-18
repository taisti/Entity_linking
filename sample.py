import csv
from pathlib import Path

import prodigy
import spacy
from brat_parser import get_entities_relations_attributes_groups


def read_entities():
    entities_list, relations_list = [], []
    datapath = Path.cwd() / "data"
    for i in range(300):
        entities, relations, _, _ = get_entities_relations_attributes_groups(datapath / "{0}.ann".format(i))
        entities_list.append(entities)
        relations_list.append(relations)

    return entities_list


def read_file_as_stream_elem(path):
    with open(path, 'r') as input_data:
        return [{"text": input_data.read()}]


@prodigy.recipe(
    "entity_linker.manual",
    dataset=("The dataset to use", "positional", None, str),
    source_dir=("The source dir", "positional", None, Path),
    recipe_number=("The number of reading recipe using to file name and index of list", "positional", None, int),
    nlp_dir=("Path to the NLP model with a pretrained NER component", "positional", None, Path),
    kb_loc=("Path to the KB", "positional", None, Path),
    entity_loc=("Path to the file with additional information about he entities", "positional", None, Path),
)
def entity_linker_manual(dataset, source_dir, recipe_number, nlp_dir, kb_loc, entity_loc):
    nlp = spacy.load(nlp_dir)
    kb = spacy.kb.KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk(kb_loc)
    # model = EntityRecognizer(nlp)

    entities_list = read_entities()

    id_dict = {}
    with entity_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            id_dict[row[0]] = (row[1], row[2])

    file_name = r"{0}.txt".format(recipe_number)
    stream = read_file_as_stream_elem(Path(source_dir, file_name))
    stream = [prodigy.util.set_hashes(eg) for eg in stream]
    # stream = (eg for score, eg in model(stream))
    for i, elem in enumerate(stream):
        spans = []
        for entity in entities_list[recipe_number].values():
            if entity.text in elem['text']:
                span = {'start': entity.span[0][0], 'end': entity.span[0][1], 'text': entity.text, 'label': entity.type}
                spans.append(span)
        stream[i]['spans'] = spans

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

            candidates = []
            for elem in mention.split(' '):
                res = kb.get_alias_candidates(elem)
                if res:
                    candidates.append(res[0])

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
    option = "<a href='" + url_prefix + entity_id + "'>" + entity_id + "</a>: " + name

    if descr and descr != 'NONE':
        option = option + ':' + descr

    return option
