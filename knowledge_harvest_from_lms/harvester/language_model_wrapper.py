import logging
import string
from copy import deepcopy

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from knowledge_harvest_from_lms.data_utils import stopword_list
from knowledge_harvest_from_lms.data_utils.data_utils import (
    find_sublist,
    get_n_ents,
    get_sent,
)


class LanguageModelWrapper:
    _device = None

    def __init__(self, model_name: str, device: str):
        self.logger = logging.getLogger(__name__)
        self._model_name = model_name
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self.device)
        self._banned_ids = None
        self._get_banned_ids()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: str):
        """set the device based on user input"""
        # TODO: Detect hardware?
        erm = "Expected a string: 'cpu', 'mps' or 'cuda'"
        allowed_devices = {'cpu', 'cuda', 'mps'}
        if not isinstance(device, str):
            raise TypeError(erm)
        device = device.lower()
        if not device in allowed_devices:
            raise ValueError(erm)
        self._device = torch.device(device)
        self.logger.debug(f'THE DEVICE IS SET TO {self._device}')

    def _get_banned_ids(self):
        self._banned_ids = self._tokenizer.all_special_ids
        for idx in range(self._tokenizer.vocab_size):
            if self._tokenizer.decode(idx).lower().strip() in stopword_list:
                self._banned_ids.append(idx)

    def get_mask_logits(self, input_text):
        with torch.no_grad():
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

        return outputs.logits[inputs['input_ids'] == self.tokenizer.mask_token_id]

    def fill_ent_tuple_in_prompt(self, prompt, ent_tuple):
        try:
            assert get_n_ents(prompt) == len(ent_tuple)
        except AssertionError as e:
            self.logger.exception(e)

        ent_tuple = deepcopy(ent_tuple)
        for ent_idx, ent in enumerate(ent_tuple):
            if prompt.startswith(f'<ENT{ent_idx}>'):
                ent_tuple[ent_idx] = ent.capitalize()

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        mask_spans = self.get_mask_spans(prompt=prompt, ent_tuple=ent_tuple)

        mask_positions = []
        for mask_span in mask_spans:
            mask_positions.extend([pos for pos in range(*mask_span)])

        masked_inputs = self.tokenizer(
            [sent] * len(mask_positions), return_tensors='pt'
        ).to(self.device)
        label_token_ids = []
        for i, pos in enumerate(mask_positions):
            label_token_ids.append(masked_inputs['input_ids'][i][pos].item())
            masked_inputs['input_ids'][i][
                mask_positions[i:]
            ] = self.tokenizer.mask_token_id

        with torch.no_grad():
            logits = self.model(**masked_inputs).logits
            logprobs = torch.log_softmax(logits, dim=-1)

        mask_logprobs = logprobs[
            torch.arange(len(mask_positions)), mask_positions, label_token_ids
        ].tolist()

        torch.cuda.empty_cache()
        self.logger.debug(
            f"""INPUT_IDS{self.tokenizer.encode(sent)}
                          MASKSPANS:{mask_spans}
                          MASK_POSITIONS:{mask_positions}
                          MASK_LOGPROBS:"{mask_logprobs}\
            """
        )
        return {
            'input_ids': self.tokenizer.encode(sent),
            'mask_spans': mask_spans,
            'mask_positions': mask_positions,
            'mask_logprobs': mask_logprobs,
        }

    def get_mask_spans(self, prompt, ent_tuple):
        assert get_n_ents(prompt) == len(ent_tuple)

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        input_ids = self._tokenizer.encode(sent)

        mask_spans = []
        for ent_idx, ent in enumerate(ent_tuple):
            prefix = prompt[: prompt.find(f'<ENT{ent_idx}>')].strip()
            for i in range(len(ent_tuple)):
                prefix = prefix.replace(f'<ENT{i}>', ent_tuple[i])
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)

            # processing -ing, -s, etc.
            ent_in_sent = prompt[prompt.find(f'<ENT{ent_idx}>') :].split()[0]
            for punc in string.punctuation:
                if punc not in '<>':
                    ent_in_sent = ent_in_sent.split(punc)[0]
            ent_in_sent = ent_in_sent.replace(f'<ENT{ent_idx}>', ent)

            # only mask the first word in an entity to
            # encourage entities with multiple words
            ent_in_sent = ent_in_sent.split()[0]

            ent_token_ids = self.tokenizer.encode(
                f' {ent_in_sent}' if sent[len(prefix)] == ' ' else ent_in_sent,
                add_special_tokens=False,
            )

            if len(prefix_ids) > 0:
                l = find_sublist(input_ids, prefix_ids) + len(prefix_ids)
            else:
                l = find_sublist(input_ids, ent_token_ids)
            r = l + len(ent_token_ids)

            assert input_ids[l:r] == ent_token_ids
            mask_spans.append([l, r])

        return mask_spans

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def banned_ids(self):
        return self._banned_ids
