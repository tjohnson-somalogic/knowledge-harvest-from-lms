import json
import os
import random
import sys
from pathlib import Path

import fire
from prettytable import PrettyTable

# TODO: Need configurable paths for I/O


def match_sources(input_path: Path) -> Path:
    """Match the relation source name from the input path return a ./relation_info/{relations_source}.json path."""
    path_parts = set(input_path.parts)
    relation_info_dir = Path(__file__).parent
    relation_info_dir = relation_info_dir / 'relation_info'
    for f_ in os.listdir(relation_info_dir):
        file_ = Path(f_)
        if os.path.isfile(relation_info_dir / file_):
            if file_.stem in path_parts:
                return relation_info_dir / file_
    raise ValueError(
        f'The input path cannot be matched to a file in {relation_info_dir}'
    )


def main(result_dir, n_present=20):
    result_dir = Path(result_dir)
    rel_set = match_sources(result_dir)
    relation_info = json.load(open(rel_set))

    summary_file = open(f'{result_dir}/summary.txt', 'w')

    for rel, info in relation_info.items():
        columns = {'Seed samples': info['seed_ent_tuples']}

        if not os.path.exists(f'{result_dir}/{rel}/ent_tuples.json'):
            print(f'outputs of relation \"{rel}\" not found. skipped.')
            continue

        weighted_prompts = json.load(open(f'{result_dir}/{rel}/prompts.json'))
        weighted_ent_tuples = json.load(open(f'{result_dir}/{rel}/ent_tuples.json'))

        if len(weighted_ent_tuples) == 0:
            print(f'outputs of relation \"{rel}\" not found. skipped.')
            continue
        weighted_ent_tuples = weighted_ent_tuples[:200]

        columns[f'Ours (Top {n_present})'] = [
            str(ent_tuple) for ent_tuple, _ in weighted_ent_tuples[:n_present]
        ]

        columns[f'Ours (Random samples over top 200 tuples)'] = [
            str(ent_tuple)
            for ent_tuple, _ in random.sample(weighted_ent_tuples, n_present)
        ]

        table = PrettyTable()
        for key, col in columns.items():
            if len(col) < n_present:
                col.extend(['\\'] * (n_present - len(col)))
            table.add_column(key, col)

        def _print_results(output_file):
            print(f'Relation: {rel}', file=output_file)
            print('Prompts:', file=output_file)
            for prompt, weight in weighted_prompts:
                print(f'- {weight:.4f} {prompt}', file=output_file)
            print('Harvested Tuples:', file=output_file)
            print(table, file=output_file)
            print('=' * 50, file=output_file, flush=True)

        # TODO Create configurable output:
        _print_results(output_file=summary_file)
        _print_results(output_file=sys.stdout)

    print(f'This summary has been saved into {summary_file.name}.')


def fire_present_result():
    fire.Fire(main)


if __name__ == '__main__':
    fire.Fire(main)
