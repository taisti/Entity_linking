# Entity_linking

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
