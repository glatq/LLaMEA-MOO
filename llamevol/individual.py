import uuid
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Individual(BaseModel):
    """
    Represents a candidate solution (an individual) in the evolutionary algorithm.
    Each individual has properties such as solution code, fitness, feedback, and metadata for additional information.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    solution: str = ""
    """The solution (code) of the individual."""
    name: str = ""
    """The name of the individual (typically the class name in the solution)."""
    description: str = ""
    """A short description of the individual (e.g., algorithm's purpose or behavior)."""
    configspace: Any = None
    """Optional configuration space for HPO."""
    generation: int = 0
    """The generation this individual belongs to."""
    fitness: Optional[float] = None
    feedback: str = ""
    error: str = ""
    parent_id: Optional[list[str]] = None
    """UUIDs of the parent individual(s)."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Dictionary to store additional metadata."""
    mutation_prompt: Optional[str] = None
    island_index: Optional[int] = None

    def set_mutation_prompt(self, mutation_prompt):
        """
        Sets the mutation prompt of this individual.

        Args:
            mutation_prompt (str): The mutation instruction to apply to this individual.
        """
        self.mutation_prompt = mutation_prompt

    def add_metadata(self, key, value):
        """
        Adds key-value pairs to the metadata dictionary.

        Args:
            key (str): The key for the metadata.
            value (Any): The value associated with the key.
        """
        self.metadata[key] = value

    def get_metadata(self, key):
        """
        Get a metadata item from the dictionary.

        Args:
            key (str): The key for the metadata to obtain.
        """
        return self.metadata.get(key)

    def set_scores(self, fitness, feedback="", error=""):
        self.fitness = fitness
        self.feedback = feedback
        self.error = error

    def get_summary(self):
        """
        Returns a string summary of this individual's key attributes.

        Returns:
            str: A string representing the individual in a summary format.
        """
        return f"{self.name}: {self.description} (Score: {self.fitness})"

    def copy(self):
        """
        Returns a copy of this individual, with a new unique ID and a reference to the current individual as its parent.

        Returns:
            Individual: A new instance of Individual with the same attributes but a different ID.
        """
        new_individual = Individual(
            solution=self.solution,
            name=self.name,
            description=self.description,
            configspace=self.configspace,
            generation=self.generation + 1,
            parent_id=[self.id],
        )
        new_individual.metadata = self.metadata.copy()
        return new_individual

    def __to_json__(self):
        """Used by LogggerJSONEncoder in utils.py."""
        return self.model_dump(mode="python")
