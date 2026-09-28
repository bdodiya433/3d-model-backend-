from abc import ABC, abstractmethod
from typing import Any


class AIModel(ABC):
    """
    Abstract Base Class representing a hot-swappable AI Model.
    All model providers must inherit from this class to guarantee a unified interface.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the model weights into memory (RAM/MPS/CUDA).
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model weights from memory and free hardware resources.
        """
        pass

    @abstractmethod
    def generate(self, input_data: Any, **kwargs: Any) -> Any:
        """
        Execute AI inference on the model using the provided input data.
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Returns True if the model weights are currently resident in memory.
        """
        pass

    @abstractmethod
    def health(self) -> bool:
        """
        Performs a self-check or check on the weights file and returns True if healthy.
        """
        pass
