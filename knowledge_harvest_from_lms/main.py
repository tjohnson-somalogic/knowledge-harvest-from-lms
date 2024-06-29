import json
import logging
import os
from pathlib import Path

import fire

from knowledge_harvest_from_lms.harvester.knowledge_harvester import KnowledgeHarvester

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG
)


def main(
    rel_set='conceptnet',
    model_name='roberta-large',
    device='cpu',
    max_n_ent_tuples=1000,
    max_n_prompts=20,
    prompt_temp=2.0,
    max_word_repeat=5,
    max_ent_subwords=2,
    use_init_prompts=False,
):
    knowledge_harvester = KnowledgeHarvester(
        model_name=model_name,
        device=device,
        max_n_ent_tuples=max_n_ent_tuples,
        max_n_prompts=max_n_prompts,
        max_word_repeat=max_word_repeat,
        max_ent_subwords=max_ent_subwords,
        prompt_temp=prompt_temp,
    )

    relation_path = Path(__file__).parent / "relation_info"
    results_path = Path(__file__).parent / "results"

    relation_info = json.load(open(relation_path / f'{rel_set}.json'))

    for rel, info in relation_info.items():
        logger.info(f'Harvesting for relation {rel}...')

        setting = f'{max_n_ent_tuples}tuples'
        if use_init_prompts:
            setting += '_initprompts'
        else:
            setting += f'_top{max_n_prompts}prompts'

        output_dir = results_path / rel_set / setting / model_name
        rel_dir = output_dir / rel
        if os.path.exists(rel_dir / "ent_tuples.json"):
            logger.info(f'file {rel_dir / "ent_tuples.json"} exists, skipped.')
            continue
        else:
            os.makedirs(rel_dir, exist_ok=True)
            json.dump([], open(rel_dir / "ent_tuples.json", 'w'))

        knowledge_harvester.clear()
        knowledge_harvester.set_seed_ent_tuples(seed_ent_tuples=info['seed_ent_tuples'])
        knowledge_harvester.set_prompts(
            prompts=info['init_prompts']
            if use_init_prompts
            else list(set(info['init_prompts'] + info['prompts']))
        )
        knowledge_harvester.update_prompts()
        prompt_output_path = rel_dir / 'prompts.json'
        json.dump(
            knowledge_harvester.weighted_prompts,
            open(prompt_output_path, 'w'),
            indent=4,
        )
        logger.debug(f'prompts written to file {prompt_output_path}')
        for prompt, weight in knowledge_harvester.weighted_prompts:
            logger.info(f'{weight:.4f} {prompt}')

        knowledge_harvester.update_ent_tuples()
        ent_tuples_output_path = rel_dir / 'ent_tuples.json'
        json.dump(
            knowledge_harvester.weighted_ent_tuples,
            open(ent_tuples_output_path, 'w'),
            indent=4,
        )
        logger.info(f'ent_tuples written to file {ent_tuples_output_path}')


def run_main_cli():
    fire.Fire(main)


if __name__ == '__main__':
    fire.Fire(main)
