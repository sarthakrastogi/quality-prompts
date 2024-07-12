# Quality Prompts
## Use and evaluate prompting techniques quickly.

Quality Prompts implements 58 prompting techniques explained in [this survey from OpenAI, Microsoft, et al.](https://arxiv.org/pdf/2406.06608) 

### 1. Install Quality Prompts:

`pip install quality-prompts`

### 2. Write the components of your prompt

```
from quality_prompts.prompt import QualityPrompt

directive = "You are given a document and your task..."
additional_information = "In the knowledge graph, ..."
output_formatting = "You will respond with a knowledge graph in..."

prompt = QualityPrompt(
                        directive,
                        additional_information,
                        output_formatting,
                        exemplar_store
                       )
```

### 3. Quality Prompts searches and uses only the few-shot examples that are relevant to the user's query

```
input_text = "list the disorders included in cvd"
prompt.few_shot(input_text=input_text, n_shots=1)
```

### 4. Simply call one of several prompting techniques to your prompt

##### System2Attention
Helps clarify the given context as an additinoal step before it's used to answer the question

```
prompt.system2attention(input_text)
print(prompt.compile())
```

```
>> You are given a document and your task is to create a knowledge graph from it.
        
In the knowledge graph, entities such as people, places, objects, institutions, topics, ideas, etc. are represented as nodes.
Whereas the relationships and actions between them are represented as edges.

Example input: Cardiovascular disease (CVD) encompasses a spectrum of...
Example output: [{'entity': 'cardiovascular disease (cvd)', 'connections': ...

You will respond with a knowledge graph in the given JSON format:

[
    {"entity" : "Entity_name", "connections" : [
        {"entity" : "Connected_entity_1", "relationship" : "Relationship_with_connected_entity_1},
        {"entity" : "Connected_entity_2", "relationship" : "Relationship_with_connected_entity_2},
        ]
    },
]
```

#### Tabular Chain of Thought Prompting
Prompts the LLM to think step by step and write the step, process and result of each step in a markdown table.
Significantly boosts accuracy in solving math problems.
```
prompt.tabular_chain_of_thought_prompting(input_text=input_text)
print(prompt.compile())
```

```
Solve the given math problem.
Think through the problem step by step to solve it.
At each step, you have to figure out:
- the step number,
- the sub-question to be answered in that step,
- the thought process of solving that step, and
- the result of solving that step.
Respond in the following markdown table format for each step:
|step|subquestion|process|result|    
```

### 6. Upcoming: Easily evaluate different prompting techniques

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sarthakrastogi/quality-prompts&type=Date)](https://star-history.com/#sarthakrastogi/quality-prompts&Date)

To stay updated on the latest features and prompting techniques added to the library, you can star this repo.