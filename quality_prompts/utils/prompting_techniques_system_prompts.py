from pydantic import BaseModel
from typing import List, Dict


class System2AttentionSystemPrompt(BaseModel):
    # Source: Page 4 of https://arxiv.org/pdf/2311.11829 -- only paragraph 1 of their prompt is used
    additional_information: str
    input_text: str

    @property
    def system_prompt(self) -> str:
        return f"""Given the following text by a user, extract the part that is unbiased and not their opinion,
    so that using that text alone would be good context for providing an unbiased answer to
    the question portion of the text.
    Text sent by User:
    {self.additional_information}
    """

    @property
    def messages(self) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": self.input_text},
        ]


class SimtoMCharacterExtractionSystemPrompt(BaseModel):
    input_text: str

    @property
    def system_prompt(self) -> str:
        return """Which character's perspective is relevant to answer this user's question."""

    @property
    def messages(self) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self.input_text,
            },
        ]


class SimtoMSystemPrompt(BaseModel):
    # Source: Page 4 of https://arxiv.org/pdf/2311.10227
    additional_information: str
    character_name: str

    @property
    def system_prompt(self) -> str:
        return f"""The following is a sequence of events:
    {self.additional_information}
    Which events does {self.character_name} know about?"""

    @property
    def messages(self) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ]


class SelfAskSystemPrompt(BaseModel):
    # Source: Written by @sarthakrastogi
    input_text: str
    additional_information: str

    @property
    def system_prompt(self) -> str:
        return f"""Given the below context and the user's question, decide whether follow-up questions are required to answer the question.
    Only if follow-up questions are absolutely required before answering the present questions, return the questions as a Python list:
    # Example response:
    ["follow_up_question_1", "follow_up_question_2"]

    If the user's question can be answered without follow up questions, simply respond with "FALSE".

    # Provided information:
    {self.additional_information}"""

    @property
    def messages(self) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self.input_text,
            },
        ]
