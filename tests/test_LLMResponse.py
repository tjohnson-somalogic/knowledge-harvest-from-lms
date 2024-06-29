import pytest

from knowledge_harvest_from_lms.models import LLMResponse

good_dict_gpt = {"choices": [{"text": "This is handled."}]}
good_dict_llama = {"response": "This is handled."}
bad_dict = {"other": "Not handled automatically."}


def test_LLMResponse_gpt():
    r = LLMResponse.from_dict(good_dict_gpt)
    assert r.text == "This is handled."


def test_LLMResponse_llama():
    r = LLMResponse.from_dict(good_dict_llama)
    assert r.text == "This is handled."


def test_LLMResponse_bad_input():
    with pytest.raises(AttributeError):
        r = LLMResponse.from_dict(bad_dict)
        r.text


def test_LLMResponse_text_explicitly_set():
    r = LLMResponse(text=bad_dict["other"])
    assert r.text == "Not handled automatically."
