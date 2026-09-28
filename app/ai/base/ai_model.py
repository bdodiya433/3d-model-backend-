from abc import ABC, abstractmethod
from typing import Any


class AIModel(ABC):
    """
    Abstract interface for swappable AI model providers.
    Ensures all AI models implement standard load, unload, and inference lifecycle methods.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Loads weights into memory and sends the model to the target hardware device.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unloads weights from memory to release VRAM/RAM resources.
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Returns whether the model is currently resident in memory.
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any, **kwargs: Any) -> Any:
        """
        Performs forward pass/inference on the loaded model.
        """
        pass
