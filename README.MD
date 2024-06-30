# Write Quality Prompts
## Use and evaluate prompting techniques [surveyed by Openai and Microsoft](https://arxiv.org/pdf/2406.06608) quickly.

### 1. Install Quality Prompts:

`pip install quality-prompts`

### 2. Write down few-shot examples:

```
from quality_prompts.prompt import QualityPrompt
from quality_prompts.exemplars import ExemplarStore, Exemplar
from quality_prompts.utils.llm import get_embedding

exemplars = [Exemplar(input="Cardiovascular disease (CVD) encompasses..."
                      label="[{"entity": "cardiovascular disease (cvd)..."
                      input_embedding=[-0.008424, 0.097374, ...]
                     ),
                     ...
            ]
exemplar_store = ExemplarStore(exemplars=exemplars)
```

### 3. Write the components of your prompt

```
directive = "You are given a document and your task..."
additional_information = "In the knowledge graph, entities..."
output_formatting = "You will respond with a knowledge graph in the given JSON"

prompt = QualityPrompt(
                        directive=directive,
                        additional_information=additional_information,
                        output_formatting=output_formatting,
                        exemplar_store=exemplar_store
                       )
```

### 4. Effortlessly add few-shot examples to your prompt and compile it.

```
input_text = "list the disorders included in cvd"
prompt.few_shot(input_text=input_text, n_shots=2)
print(prompt.compile())
```

```
>> You are given a document and your task is to create a knowledge graph from it.
        
In the knowledge graph, entities such as people, places, objects, institutions, topics, ideas, etc. are represented as nodes.
Whereas the relationships and actions between them are represented as edges.

Example input: Cardiovascular disease (CVD) encompasses a spectrum of...
Example output: [{'entity': 'cardiovascular disease (cvd)', 'connections': ...

Example input: The epidemiological burden of cardiovascular disease underscores ...
You will respond with a knowledge graph in the given JSON format:

[
    {"entity" : "Entity_name", "connections" : [
        {"entity" : "Connected_entity_1", "relationship" : "Relationship_with_connected_entity_1},
        {"entity" : "Connected_entity_2", "relationship" : "Relationship_with_connected_entity_2},
        ]
    },
]
```
