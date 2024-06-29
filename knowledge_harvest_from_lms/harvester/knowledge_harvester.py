import logging

from scipy.special import softmax
from tqdm import tqdm

from knowledge_harvest_from_lms.data_utils.data_utils import (
    fix_prompt_style,
    is_valid_prompt,
)

from .entity_tuple_searcher import EntityTupleSearcher
from .language_model_wrapper import LanguageModelWrapper


class KnowledgeHarvester:
    def __init__(
        self,
        model_name,
        max_n_prompts=20,
        max_n_ent_tuples=10000,
        max_word_repeat=5,
        max_ent_subwords=1,
        prompt_temp=1.0,
        device="cpu",
    ):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._max_n_prompts = max_n_prompts
        self._max_n_ent_tuples = max_n_ent_tuples
        self._max_word_repeat = max_word_repeat
        self._max_ent_subwords = max_ent_subwords
        self._prompt_temp = prompt_temp
        self._model = LanguageModelWrapper(model_name=model_name, device=device)
        self._ent_tuple_searcher = EntityTupleSearcher(model=self._model)
        self.logger = logging.getLogger(__name__)
        self._seed_ent_tuples = None

    def clear(self):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._seed_ent_tuples = None

    def set_seed_ent_tuples(self, seed_ent_tuples):
        self.logger.debug('Setting Seed Ent Tuples')
        self._seed_ent_tuples = seed_ent_tuples
        self.logger.debug(f'{len(self._seed_ent_tuples)} tuples set.')

    def set_prompts(self, prompts):
        self.logger.debug('Setting Propts')
        for prompt in prompts:
            if is_valid_prompt(prompt=prompt):
                self._weighted_prompts.append([fix_prompt_style(prompt), 1.0])
        self.logger.debug(f'{len(self._weighted_prompts)} prompts set!')

    def update_prompts(self):
        self.logger.debug('Updating Prompts')
        for i, (prompt, _) in enumerate(self._weighted_prompts):
            pos_scores, neg_scores = [], []
            for ent_tuple in self._seed_ent_tuples:
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
                self.logger.debug(f'PROMPT: {prompt}\nENT_TUPLE:{ent_tuple}')
                pos_scores.append(self.score(prompt=prompt, ent_tuple=ent_tuple))

                for ent_idx in range(len(ent_tuple)):
                    for ent_tuple1 in self._seed_ent_tuples:
                        if ent_tuple1[ent_idx] == ent_tuple[ent_idx]:
                            continue

                        ent_tuple_neg = (
                            ent_tuple[:ent_idx]
                            + [ent_tuple1[ent_idx]]
                            + ent_tuple[ent_idx + 1 :]
                        )

                        neg_scores.append(
                            self.score(prompt=prompt, ent_tuple=ent_tuple_neg)
                        )

            pos_score = sum(pos_scores) / len(pos_scores)
            neg_score = sum(neg_scores) / len(neg_scores)

            self._weighted_prompts[i][1] = (
                pos_score - 0.5 * neg_score
            ) / self._prompt_temp

        self._weighted_prompts = sorted(
            self._weighted_prompts, key=lambda t: t[1], reverse=True
        )[: self._max_n_prompts]
        self.logger.debug('Prompts Updated, Updating Weights')
        norm_weights = softmax([weight for _, weight in self._weighted_prompts])
        norm_weights[norm_weights < 0.05] = 0.0
        norm_weights /= norm_weights.sum()

        for i, norm_weight in enumerate(norm_weights):
            self._weighted_prompts[i][1] = norm_weight
        self._weighted_prompts = [t for t in self._weighted_prompts if t[1] > 1e-4]
        self.logger.debug('Weights Updated')

    def update_ent_tuples(self):
        self.logger.debug('Updating Entity Tuples')
        ent_tuples = self._ent_tuple_searcher.search(
            weighted_prompts=self._weighted_prompts,
            n=self._max_n_ent_tuples,
            max_word_repeat=self._max_word_repeat,
            max_ent_subwords=self._max_ent_subwords,
        )

        self._weighted_ent_tuples = []
        for ent_tuple in tqdm(ent_tuples, desc='re-scoring ent_tuples'):
            best_ent_tuple = None
            best_score = float('-inf')
            for t in range(1 << len(ent_tuple)):
                bin_code = f'{t:b}'
                bin_code = '0' * (len(ent_tuple) - len(bin_code)) + bin_code

                coded_ent_tuple = []
                for b, ent in zip(bin_code, ent_tuple):
                    coded_ent_tuple.append(ent.title() if b == '1' else ent)

                score = self.score_ent_tuple(ent_tuple=coded_ent_tuple)
                if score > best_score:
                    best_score = score
                    best_ent_tuple = coded_ent_tuple

            self._weighted_ent_tuples.append([best_ent_tuple, best_score])

        self._weighted_ent_tuples = sorted(
            self._weighted_ent_tuples, key=lambda t: t[1], reverse=True
        )
        self.logger.debug('Updating ENT_TUPLE weights')
        norm_weights = softmax([weight for _, weight in self._weighted_ent_tuples])
        for i, norm_weight in enumerate(norm_weights):
            self._weighted_ent_tuples[i][1] = norm_weight
        self.logger.debug('Completed End Tuple')

    def score_ent_tuple(self, ent_tuple):
        score = 0.0
        for prompt, weight in self.weighted_prompts:
            score += weight * self.score(prompt=prompt, ent_tuple=ent_tuple)
        self.logger.debug(f'PROMPT:{prompt}\nWEIGHT:{weight}\nSCORE:{score}')
        return score

    def score(self, prompt, ent_tuple):
        self.logger.debug('Scoring prompt.')
        logprobs = self._model.fill_ent_tuple_in_prompt(
            prompt=prompt, ent_tuple=ent_tuple
        )['mask_logprobs']
        self.logger.debug(f'LOGPROBS {logprobs}')
        token_wise_score = sum(logprobs) / len(logprobs)
        ent_wise_score = sum(logprobs) / len(ent_tuple)
        min_score = min(logprobs)
        score = (token_wise_score + ent_wise_score + min_score) / 3.0
        self.logger.debug(f"SCORE: {score}")
        return score

    @property
    def weighted_ent_tuples(self):
        return self._weighted_ent_tuples

    @property
    def weighted_prompts(self):
        return self._weighted_prompts
