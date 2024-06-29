from abc import ABC, abstractmethod


class LLMResource(ABC):
    """An abstract class for an LLM resource which should at minimum include a call() method."""

    def __init__(self):
        pass

    @abstractmethod
    def call():
        raise NotImplementedError(
            "LLMResource.call must be overridden in child classes"
        )


class LLMResponse:
    """The LLMResponse object consumes a response dictionary and returns a python object with a LLM.text attribute which returns the model's text response.

    Certain key features like "response" or x["choices"][0]["message"]["content"] will automatically display in the cls.text attribute. Others will need to be explicitly set.
    """

    _text = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def text(self):
        attrs = set([x for x in self.__dict__.keys()])
        if self._text:
            # if the user set cls.text return the value.
            return self._text
        elif "choices" in attrs:
            choice = self.choices[0]
            for x in ("text", "message"):
                # current and legacy OpenAI keys
                if x in choice.keys():
                    self._text = choice[x]
                    return self._text
        elif "response" in attrs:
            # The ollama response
            return self.response
        else:
            raise AttributeError(
                "The model response is not recognized. You may need to set the value of cls.text explicitly"
            )

    @text.setter
    def text(self, string: str):
        if not isinstance(string, str):
            raise TypeError(f"Expected a string response not {type(string)}")
        self._text = string

    @text.deleter
    def text(self):
        self._text = None

    @classmethod
    def from_dict(cls, dictionary):
        """It is recommended to generate this class from dictionaries using this method."""
        return cls(**dictionary)
