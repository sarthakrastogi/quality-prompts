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

    def format(self):
        return f"""Input: {self.input}
        Output: {self.label}"""


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
        input_embedding = get_embedding(input_text)
        input_embedding = np.array(input_embedding).reshape(1, -1)

        # Extract embeddings of all exemplars
        example_embeddings = np.array(
            [example.input_embedding for example in self.exemplars]
        )

        if exemplar_selection_method == "knn":
            difficult_exemplars = [
                ex for ex in self.exemplars if ex.complexity_level == "high"
            ]
            medium_and_simple_exemplars = [
                ex for ex in self.exemplars if ex.complexity_level in ["medium", "low"]
            ]

            if prioritise_complex_exemplars:
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
                    if difficult_embeddings.size == 0:
                        raise ValueError("No difficult exemplars found.")

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
                example_embeddings = np.array(
                    [example.input_embedding for example in exemplars_to_search]
                )

            # Ensure example_embeddings is not empty
            if example_embeddings.size == 0:
                raise ValueError("No exemplars found for KNN search.")

            # Initialize and fit the NearestNeighbors model
            nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
            nbrs.fit(example_embeddings)
            distances, indices = nbrs.kneighbors(input_embedding)

            # Return the top k closest exemplars
            return [exemplars_to_search[i] for i in indices.flatten()]

        elif exemplar_selection_method == "vote-k":
            pass  # TODO

        elif exemplar_selection_method == "sg-icl":
            pass  # TODO
