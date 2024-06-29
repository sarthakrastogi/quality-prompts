from pydantic import BaseModel
from .utils.llm import llm_call
from .exemplars import ExemplarStore
import warnings


class QualityPrompt(BaseModel):
    directive: str  # Core intent of the prompt
    output_formatting: str
    style_instructions: str = ""
    role_instructions: str = ""
    emotion_instructions: str = ""
    additional_information: str
    exemplar_store: ExemplarStore = ExemplarStore(exemplars=[])

    def few_shot(self, input_text, n_shots=3):
        if len(self.exemplar_store.exemplars) > n_shots:
            shots = self.exemplar_store.knn(input_text=input_text, k=n_shots)
        else:
            shots = self.exemplar_store.exemplars
        return shots

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
