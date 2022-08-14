# Entity_linking
In order to run entity linker, run the following steps:
```
cd entity_linker
pip3 install -r requirements.txt
python3 entity_linker.py
```

The output of the process is stored in a CSV file that joins each BRAT annotation with its metadata and appropriate entity IRI and label.

The output file consists of the following columns:
```
    file_id  - the numeric id of the BRAT annotation file (eg. 100)
    id       - annotation id in BRAT annotation file (eg., T1 marking the first token)
    category - the category a given span was assigned by a linguist
    start    - where the span marked in BRAT begins
    end      - where the span marked in BRAT ends
    text     - the span text itself
    iri      - the linked entity IRI (NONE if nothing linked)
    label    - the linked entity LABEL (NONE if nothing is linked)

```

The `entity_linker.py` script expects 3 optional parameters:
```
    --ontology_path - Path to an ontology we want to link to (by default it is set to ../foodon.owl)
    --annotations_path - Path to a folder with BRAT annotations (by default it is set to ../data)
    --output_file_path - Path to a result CSV file (by default it is set to ./report.csv)
```


# Prodigy


To run some example code, you can simply type:
`bash run.sh`

It contains the following lines:
```
pip3 install nltk
python3 generate_kb.py
prodigy entity_linker.manual sandbox data 3 output_food/my_nlp output_food/my_kb food_product_entities.csv -F sample.py
``` 
the most important of which is the last one, running prodigy with the following arguments:

```entity_linker.manual - type of task
sandbox - unused at the moment, can be any identifier
data - folder where our dataset is stored
3 - the id of a file in data folder (recipe number stored in filename)
output_food/my_nlp - Path to NLP model
output_food/my_kb  - Path to KB
food_product_entities.csv - Path to the file with additional information about entities
```
