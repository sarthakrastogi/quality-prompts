from pydantic import BaseModel
from typing import List, Dict


class System2AttentionSystemPrompt(BaseModel):
    # Source: Page 4 of https://arxiv.org/pdf/2311.11829
    additional_information: str
    input_text: str
    system_prompt: str = f"""Given the following text by a user, extract the part that is unbiased and not their opinion,
so that using that text alone would be good context for providing an unbiased answer to
the question portion of the text.
Please include the actual question or query that the user is asking. Separate this
into two categories labeled with “Unbiased text context (includes all content except user’s
bias):” and “Question/Query (does not include user bias/preference):”.
Text by User:
{additional_information}
"""
    messages: List[Dict[str:str]] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": input_text},
    ]


class SimtoMCharacterExtractionSystemPrompt(BaseModel):
    input_text: str
    system_prompt: str = (
        """Which character's perspective is relevant to answer this user's question."""
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]


class SimtoMSystemPrompt(BaseModel):
    # Source: Page 4 of https://arxiv.org/pdf/2311.10227
    additional_information: str
    character_name: str
    system_prompt: str = f"""The following is a sequence of events:
    {additional_information}
    Which events does {character_name} know
    about?"""
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]


class SelfAskSystemPrompt(BaseModel):
    # Source: Written by @sarthakrastogi
    input_text: str
    additional_information: str
    system_prompt: str = f"""Given the below context and the user's question, decide whether follow-up questions are required to answer the question.
    Only if follow-up questions are absolutely required before answering the present questions, return the questions as a Python list:
    # Example response:
    ["follow_up_question_1", "follow_up_question_2"]

    If the user's question can be answered without follow up questions, simply respond with "FALSE".
    
    # Provided information:
    {additional_information}"""
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": input_text,
        },
    ]
