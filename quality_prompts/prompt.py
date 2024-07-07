from pydantic import BaseModel
import warnings
from typing import List
import json

from .exemplars import ExemplarStore, Exemplar
from .utils.llm import llm_call
from .utils.prompting_techniques_system_prompts import *


class QualityPrompt(BaseModel):
    directive: str  # Core intent of the prompt
    output_formatting: str = ""
    additional_information: str = ""
    style_instructions: str = ""
    role_instructions: str = ""
    emotion_instructions: str = ""
    exemplar_store: ExemplarStore = ExemplarStore(exemplars=[])
    few_shot_examples: List[Exemplar] = []

    def compile(self):
        formatted_examples = "\n".join(
            [
                f"Example input: {e.input}\nExample output: {e.label}\n"
                for e in self.few_shot_examples
            ]
        )
        return f"""{self.directive}
        {self.additional_information}
        {formatted_examples}
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

    def system2attenton(self, input_text):
        """
        Makes an LLM rewrite the prompt by removing any info unrelated to the user's question.
        https://arxiv.org/abs/2311.11829
        """
        messages = System2AttentionSystemPrompt(
            additional_information=self.additional_information, input_text=input_text
        ).messages
        self.additional_information = llm_call(messages=messages)

    def sim_to_M(self, input_text):
        """
        Establishes the known facts
        https://arxiv.org/abs/2311.10227
        """
        messages = SimtoMCharacterExtractionSystemPrompt(input_text=input_text).messages
        character_name = llm_call(messages=messages)

        messages = SimtoMSystemPrompt(
            additional_information=self.additional_information,
            character_name=character_name,
        ).messages
        self.additional_information = llm_call(messages=messages)

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

    def self_ask(self, input_text, allow_search_engine=False):
        """
        Prompts the LLM to first ask any follow-up questions if needed
        http://arxiv.org/abs/2210.03350
        """
        messages = SelfAskSystemPrompt(
            input_text=input_text, additional_information=self.additional_information
        ).messages
        response = llm_call(messages=messages)
        if "FALSE" in response:
            pass
        else:
            follow_up_questions = json.loads(response)
            for follow_up_question in follow_up_questions:
                if allow_search_engine:
                    pass  # TODO
                else:
                    messages = [
                        {"role": "system", "content": self.additional_information},
                        {"role": "user", "content": follow_up_question},
                    ]
                    follow_up_question_answer = llm_call(messages=messages)

                self.additional_information += f"""Question: {follow_up_question}
                                                   Answer: {follow_up_question_answer}
                                                """

    def step_back_prompting(self, input_text):
        """
        Prompts the LLM to first generate generic questions about facts/concepts used to answer the question, before answering.
        https://arxiv.org/pdf/2310.06117
        """
        messages = StepBackPromptingSystemPrompt(
            input_text=input_text, additional_information=self.additional_information
        ).messages
        step_back_question = llm_call(messages=messages)

        messages = [
            {"role": "system", "content": self.additional_information},
            {"role": "user", "content": step_back_question},
        ]
        step_back_answer = llm_call(messages=messages)

        self.additional_information += f"""Question: {step_back_question}
                                            Answer: {step_back_answer}
                                        """

    def analogical_prompting(self, input_text):
        """
        Prompts the LLM to generate three distinct questions (along with solutions) with are similar to the user's query, and then finally solve the user's query.
        https://arxiv.org/pdf/2310.01714
        """
        analogical_prompting_system_prompt = AnalogicalPromptingSystemPrompt(
            input_text=input_text, directive=self.directive
        )
        self.directive, self.output_formatting = (
            analogical_prompting_system_prompt.updated_directive,
            analogical_prompting_system_prompt.updated_output_formatting,
        )

    def thread_of_thought_prompting(self, input_text):
        """
        Prompts the LLM to first analyse and summarise and additional information / context step by step, before answering.
        https://arxiv.org/pdf/2311.08734
        """
        thread_of_thought_context_summarisation_messages = (
            ThreadOfThoughtPromptingSystemPrompt(
                additional_information=self.additional_information
            ).context_summarisation_system_prompt
        ).context_summarisation_messages

        self.additional_information = llm_call(
            messages=thread_of_thought_context_summarisation_messages
        )

    def tabular_chain_of_thought_prompting(self, input_text):
        """
        Prompts the LLM to think step by step and write the step, process and result of each step in a markdown table
        https://arxiv.org/pdf/2305.17812
        """
        tabcot_prompting_system_prompt = TabularChainOfThoughtPrompingSystemPrompt(
            input_text=input_text,
            directive=self.directive,
            output_formatting=self.output_formatting,
        )
        self.directive, self.output_formatting = (
            tabcot_prompting_system_prompt.updated_directive,
            tabcot_prompting_system_prompt.updated_output_formatting,
        )
