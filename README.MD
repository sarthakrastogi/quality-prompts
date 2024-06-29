# 🌌 Nebulous AI
## Create AI agents for your own use case quickly.

AI agents are the future of automation and LLMs. Nebulous helps you create agents:
* For your own business use case
* In just 5 lines of code
* By using the best agent algorithms 


It's easy to learn and get started with. Ready to create your first agent in just 5 lines of code? [Learn how.](https://github.com/sarthakrastogi/nebulousai/tree/main?tab=readme-ov-file#create-ai-agents-with-top-performing-algorithms-quickly)

### 1. Install Nebulous AI:

`pip install nebulousai`

### 2. Create an agent:

```
from nebulousai.core.planning import Planner
from nebulousai.core.brain import Brain
from nebulousai.agent import PlannerAgent
from nebulousai.all_tools import searchWikipediaTool

brain = Brain(brain_system_prompt="Answer questions by using Wikipedia as a source.")
planner = Planner(agent_goal="find obama's hobbies")

agent = PlannerAgent(name="wiki_agent",
              brain=brain,
              planner=planner)

agent.add_tool(searchWikipediaTool)
```

### 3. Let the agent plan a task and act on it:

```
result = agent.act()
```

```

>> {'tool_name': 'Wikipedia Search', 'arguments': {'search_term': 'Barack Obama'}}
Performed 'search_wikipedia'.
>> {'tool_name': 'Reason', 'arguments': {'input_text': 'What hobbies does Barack Obama enjoy?'}}
Performed 'reason'.

>> Result: Some of Barack Obama's favorite hobbies include playing basketball, reading, writing, and spending time outdoors. He is also known to be a fan of music and enjoys playing the guitar. Additionally, he has been seen golfing and has a passion for cooking and trying new foods.
```