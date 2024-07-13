from pydantic import BaseModel
from typing import List
from sklearn.neighbors import NearestNeighbors
import numpy as np

from .utils.llm import get_embedding


class Exemplar(BaseModel):
    input: str
    label: str
    input_embedding: List[float]
    complexity_level: str = "medium"


class ExemplarStore(BaseModel):
    exemplars: List[Exemplar]

    def size(self):
        return len(self.exemplars)

    def get_similar_exemplars_to_test_sample(
        self,
        input_text,
        exemplar_selection_method="knn",
        k=3,
        prioritise_complex_exemplars=False,
    ):
        """
        If there are a large number of exemplars, the ones best-matching to the user's query can be actually included for few-shot prompting.
        https://arxiv.org/abs/2101.06804
        """

        input_embedding = get_embedding(input_text)
        input_embedding = np.array(input_embedding).reshape(1, -1)

        # Extract embeddings of all exemplars
        example_embeddings = np.array(
            [example.input_embedding for example in self.exemplars]
        )

        if exemplar_selection_method == "knn":
            if prioritise_complex_exemplars:
                # Filter exemplars by complexity level
                difficult_exemplars = [
                    ex for ex in self.exemplars if ex.complexity_level == "difficult"
                ]
                medium_and_simple_exemplars = [
                    ex
                    for ex in self.exemplars
                    if ex.complexity_level in ["medium", "simple"]
                ]

                if len(difficult_exemplars) >= k:
                    # Only use difficult exemplars for KNN
                    example_embeddings = np.array(
                        [example.input_embedding for example in difficult_exemplars]
                    )
                    exemplars_to_search = difficult_exemplars
                else:
                    # Use all difficult exemplars and fill the rest with medium and simple ones
                    difficult_embeddings = np.array(
                        [example.input_embedding for example in difficult_exemplars]
                    )
                    medium_simple_embeddings = np.array(
                        [
                            example.input_embedding
                            for example in medium_and_simple_exemplars
                        ]
                    )
                    example_embeddings = np.vstack(
                        (difficult_embeddings, medium_simple_embeddings)
                    )
                    exemplars_to_search = (
                        difficult_exemplars + medium_and_simple_exemplars
                    )
            else:
                exemplars_to_search = self.exemplars

            # Initialize and fit the NearestNeighbors model
            nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
            nbrs.fit(example_embeddings)
            distances, indices = nbrs.kneighbors(input_embedding)

            # Return the top k closest exemplars
            return [exemplars_to_search[i] for i in indices.flatten()]

        elif exemplar_selection_method == "vote-k":
            # Uses an LLM to suggest useful unlabelled exemplars
            # Also ensures that exemplars are diverse
            pass  # TODO

        elif exemplar_selection_method == "sg-icl":
            # Self-Generated In-Context Learning (SG-ICL)
            # Uses an LLM to generate exemplars when there is no training data
            pass  # TODO
