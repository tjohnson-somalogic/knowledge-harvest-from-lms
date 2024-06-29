import ollama

from . import LLMResource, LLMResponse


class OllamaModel(LLMResource):

    """Utilize a model in a locally hosted Ollama server. The model name is declared on instantiation."""

    _model = 'llama3'

    def __init__(self, model='llama3'):
        self.model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: str):
        if not isinstance(model, str):
            raise TypeError('model must be a string')
        valid_models = set([x['name'] for x in ollama.list()['models']])
        if model in valid_models:
            self._model = model
        else:
            raise ValueError(
                f'The model "{model} is not found do you need to pull it with `ollama pull {model}`?"'
            )

    def call(
        self,
        prompt: str = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 40,
        **kwargs,
    ):
        """Send a generate call to the Ollama engine. Self.model is applied automatically. Additional arguments can be supplied to the ollama model

        Parameters:
        prompt:str - prompt that will guide the models text generation.
        temperature:float - a float from low (0.001) randomness to high (2.0) randomness. Higher numbers allow more 'creativity' from the models by selecting less probable tokens.
        top_p:float flot a probability between 0.0 and 1.0
        top_k:float

        """
        # these keys are aligned with https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values at the time of writing.
        options_keys = {
            "mirostat",
            "mirostat_eta",
            "mirostat_tau",
            "num_ctx",
            "repeat_last_n",
            "repeat_penalty",
            "temperature",
            "seed",
            "stop",
            "tfs_z",
            "num_predict",
            "top_k",
            "top_p",
        }
        options = {k: v for k, v in locals().items() if k in options_keys}
        result = ollama.generate(
            model=self.model, prompt=prompt, options=options, **kwargs
        )
        return LLMResponse.from_dict(result)
