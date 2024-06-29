from pydantic import BaseModel
import warnings
from typing import List

from .utils.llm import llm_call
from .exemplars import ExemplarStore, Exemplar


class QualityPrompt(BaseModel):
    directive: str  # Core intent of the prompt
    output_formatting: str
    additional_information: str
    style_instructions: str = ""
    role_instructions: str = ""
    emotion_instructions: str = ""
    exemplar_store: ExemplarStore = ExemplarStore(exemplars=[])
    few_shot_examples: List[Exemplar] = []

    def prepare(self):
        formatted_examples = [
            f"Example input: {e.input}\nExample output: {e.label}\n"
            for e in self.few_shot_examples
        ]

        return f"""{self.directive}
        {self.additional_information}
        {"\n".join(formatted_examples)}
        {self.output_formatting}
        """

    def few_shot(self, input_text, n_shots=3):
        if len(self.exemplar_store.exemplars) > n_shots:
            self.few_shot_examples = (
                self.exemplar_store.get_similar_exemplars_to_test_sample(
                    input_text=input_text, k=n_shots
                )
            )
        else:
            self.few_shot_examples = self.exemplar_store.exemplars

    def zero_shot(self):
        shots = []
        if not self.style_instructions:
            warnings.warn("Warning: 'style_instructions' is not provided.", UserWarning)
        if not self.role_instructions:
            warnings.warn("Warning: 'role_instructions' is not provided.", UserWarning)
        if not self.emotion_instructions:
            warnings.warn(
                "Warning: 'emotion_instructions' is not provided.", UserWarning
            )
        return shots

    def system2attenton(self, input_text):
        """
        Makes an LLM rewrite the prompt by removing any info unrelated to the user's question.
        https://arxiv.org/abs/2311.11829
        """
        pass  # TODO

    def sim_to_M(self, input_text):
        """
        Establishes the known facts
        https://arxiv.org/abs/2311.10227
        """
        pass

    def rephrase_and_respond(self, input_text, perform_in="same_pass"):
        """
        http://arxiv.org/abs/2311.04205
        """
        RaR_instruction = "Rephrase and expand the question, and respond."
        if perform_in == "same_shot":
            input_text += RaR_instruction
        elif perform_in == "separate_llm_call":
            messages = [
                {"role": "system", "content": RaR_instruction},
                {"role": "user", "content": input_text},
            ]
            input_text += llm_call(messages=messages)

    def rereading(self, input_text):
        """
        http://arxiv.org/abs/2309.06275
        """
        input_text += "Read the question again:" + input_text

    def self_ask(self, input_text):
        """
        Prompts the LLM to first ask any follow-up questions if needed
        http://arxiv.org/abs/2210.03350
        """
        pass
