# Quality Prompts
## Use and evaluate prompting techniques quickly.

Quality Prompts implements 58 prompting techniques explained in [this survey from OpenAI, Microsoft, et al.](https://arxiv.org/pdf/2406.06608) 

![Quality Prompts](assets/big_dipper_design.jpg)

### 1. Install Quality Prompts:

`pip install quality-prompts`

### 2. Write the components of your prompt

```
from quality_prompts.prompt import QualityPrompt

directive = "You are given a document and your task..."
additional_information = "In the knowledge graph, ..."
output_formatting = "You will respond with a ..."

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

#### System2Attention
Helps clarify the given context as an additinoal step before it's used to answer the question

```
prompt.system2attention(input_text)
```

#### Tabular Chain of Thought
Prompts the LLM to think step by step and write the step, process and result of each step in a markdown table.
Significantly boosts accuracy in solving math problems.

```
prompt.tabular_chain_of_thought_prompting(input_text)
```

### 6. Upcoming: Easily evaluate different prompting techniques

## Star History

To stay updated on the latest evaluation features and prompting techniques added to the library, you can star this repo.

[![Star History Chart](https://api.star-history.com/svg?repos=sarthakrastogi/quality-prompts&type=Date)](https://star-history.com/#sarthakrastogi/quality-prompts&Date)

