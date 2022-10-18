# Entity_linking
In order to simply run entity linker with default params, run the following steps:
```
cd entity_linker
pip3 install -r requirements.txt
python3 entity_linker.py
```

The output of the aforementioned script is stored in a CSV file named `report.csv` that joins each BRAT annotation with its metadata and appropriate entity IRI and label.

By default this script assumes the following parameters:
- loads the `foodon.owl` ontology file located in `../foodon.owl`
- loads the BRAT annotated dataset located in `../data`
- outputs the result of entity linking to `report.csv` file.



The output file consists of the following columns:
```
    file_id  - the numeric id of the BRAT annotation file (eg. 100)
    id       - annotation id in BRAT annotation file (eg., T1 marking the first token)
    category - the category a given span was assigned by a linguist
    start    - where the span marked in BRAT begins
    end      - where the span marked in BRAT ends
    text     - the span text itself
    annotation_source - the source of an annotation, either BRAT (AnnotationSource.BRAT) or the NER output (AnnotationSource.NER)
    iri      - the linked entity IRI (NONE if nothing linked)
    label    - the linked entity LABEL (NONE if nothing is linked)

```
An example line from the output:
`220,T10,food_product_with_unit,19,28,chocolate,AnnotationSource.BRAT,http://purl.obolibrary.org/obo/FOODON_03307240,chocolate`
tells us that in the document named `220.txt` and its BRAT annotations `220.ann` the token identified as `T10` of the type `food_product_with_unit`, which spans between characters `19` and `28` and consists of text `chocolate` is linked to IRI `http://purl.obolibrary.org/obo/FOODON_03307240` that has a label `chocolate`.



The `entity_linker.py` script can be parametrized with optional arguments
```
    --ontology_path - Path to an ontology we want to link to (by default it is set to ../foodon.owl)
    --annotations_path - Path to a folder with BRAT annotations (by default it is set to ../data)
    --ner_output - If provided, it forces to process NER output stored in a given file instead of the BRAT annotated dataset.
    --output_file_path - Path to a result CSV file (by default it is set to ./report.csv)
```

For example: 
```
cd entity_linker
pip3 install -r requirements.txt
python3 entity_linker.py --ner_output ../ner_output.json --output_file_path ./NER_report.csv
```
Runs entity linker over the NER output located in `../ner_output.json` file and stores the result into `./NER_report.csv` file.

## How to run Entity Linker with NER?
- Ger NER: `git clone https://github.com/taisti/ner`
- Install requierements `pip install -r requirements.txt`
- Move into the source code `mv src`
- Train a model with `python3 train_model.py`
- Run interactive prediction over your own texts using `python3 predict.py` script. It will ask you to loop over your examples. When you finish, it will store the output of the ner in a json file (by default `output.json` file).
- Move to Entity linker folder and install dependencies using `pip install -r requirements`.
- Run entity linker (e.g., `python3 entity_linker.py --ner_output <PATH_TO_OUTPUT_JSON_FILE_GENERATED_TWO_STEPS_AGO> --output_file_path ./NER_report.csv`)
- Your report with the result should be stored in appropriate file (e.g., `./NER_report.csv` or `report.csv` by default).

# Prodigy


To run some example code, you can simply type:
`bash run.sh`

It contains the following lines:
```
pip3 install nltk
pip install mendelai-brat-parser==0.0.4
pip3 install en-core-web-lg -f https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.2.0/en_core_web_lg-3.2.0-py3-none-any.whl
python3 generate_kb.py
prodigy entity_linker.manual sandbox output_food/my_nlp output_food/my_kb food_product_entities.csv -F sample.py
``` 
the most important of which is the last one, running prodigy with the following arguments:

```entity_linker.manual - type of task
sandbox - task identifier
output_food/my_nlp - Path to NLP model
output_food/my_kb  - Path to KB
food_product_entities.csv - Path to the file with additional information about entities
```
=======

Main file: Prodigy_food.ipynb

File to gereate food product entities: create_entities_file.ipynb


Data: food_product_entities.csv

File with code to run manual annotation: sample.py


Simple Introduction to Prodigy: Prodigy_introduction.ipynb
