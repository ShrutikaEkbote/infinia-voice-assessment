from abc import ABC
from abc import abstractmethod


class BaseTTS(ABC):

    @abstractmethod
    def load_model(self):
        """
        Load the TTS model.
        """
        pass


    @abstractmethod
    def generate(
        self,
        text,
        reference_audio,
        output_path
    ):
        """
        Generate speech.
        """
        pass