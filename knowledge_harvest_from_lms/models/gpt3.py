import os

import openai

from . import LLMResource, LLMResponse


class GPT3(LLMResource):
    """Utilize GPT3 using an OpenAI API key stored in your environment under "OPENAI_API_KEY"."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def call(
        self,
        prompt,
        engine="text-davinci-002",
        temperature=1.0,
        max_tokens=30,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=0,
        n=1,
    ):
        completion = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            n=n,
        )
        return LLMResponse.from_dict(completion)
