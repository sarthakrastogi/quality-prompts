from pydantic import BaseModel
from typing import List
from sklearn.neighbors import NearestNeighbors
import numpy as np

from .utils.llm import get_embedding


class Exemplar(BaseModel):
    input: str
    label: str
    input_embedding: List[float]


class ExemplarStore(BaseModel):
    exemplars: List[Exemplar]

    def size(self):
        return len(self.exemplars)

    def get_similar_exemplars_to_test_sample(
        self, input_text, exemplar_selection_method="knn", k=3
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
            # Initialize and fit the NearestNeighbors model
            nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
            nbrs.fit(example_embeddings)
            distances, indices = nbrs.kneighbors(input_embedding)

            # Return the top k closest exemplars
            return [self.exemplars[i] for i in indices.flatten()]

        elif exemplar_selection_method == "vote-k":
            # Uses an LLM to suggest useful unlabelled exemplars
            # Also ensures that exemplars are diverse
            pass  # TODO

        elif exemplar_selection_method == "sg-icl":
            # Self-Generated In-Context Learning (SG-ICL)
            # Uses an LLM to generate exemplars when there is no training data
            pass  # TODO
