import csv
from typing import Iterable, List

from algorithms.stats import Stats
from config import Config
from utils.board import OBJECTIVE_STATE, State
from utils.node import Node


def generate_solution_yaml(tree: Iterable[Node], stats: Stats, config: Config):
    is_first = True
    init_node: Node
    intermediate_nodes = []

    for n in tree:
        if is_first:
            is_first = False
            init_node = n
        else:
            intermediate_nodes.append(n)

    solution = {
        'algorithm': config.algorithm_str,
        'solution_found': stats.objective_found,
        'objective_distance': stats.objective_distance,
        'objective_cost': stats.objective_cost,
        'explored_nodes': stats.explored_nodes_count,
        'border_nodes': stats.border_nodes_count,
        'processing_time': stats.get_processing_time(),
        'solution': {
            'init_state': init_node.__str__(),
            'intermediate_states': list(map(lambda n: n.__str__(), intermediate_nodes)),
            'final_state': OBJECTIVE_STATE.__str__()
        }
    }

    return solution


CSV_HEADERS = ['algorithm', 'solution_found', 'objective_distance', 'objective_cost', 'explored_nodes', 'border_nodes',
               'processing_time', 'heuristic', 'limit']


def _generate_csv(file_name: str, results: List[List[str]]):
    with open(file_name, 'a') as csv_file:
        writer = csv.writer(csv_file)

        if not csv_file.tell():
            writer.writerow(CSV_HEADERS)

        writer.writerows(results)


def generate_algorithm_results(algorithm: str, init_state: State, file_name=None,
                               rounds: int = 50, limit=None, heuristic=None):
    stats = Stats()
    config = Config(algorithm, limit, heuristic)

    results = []

    print(f'Calculating results for algorithm {algorithm}')

    for r in range(rounds):
        config.algorithm(init_state, stats, config)
        results.append([config.algorithm_str, stats.objective_found, stats.objective_distance, stats.objective_cost,
                        stats.explored_nodes_count, stats.border_nodes_count, stats.get_processing_time(),
                        config.heuristic_str if config.heuristic_str else "",
                        config.limit if config.limit else ""])

        if (r + 1) % 10 == 0:
            print(f'Finished {r + 1} rounds')

    print(f'Generating results csv for algorithm {algorithm}')
    _generate_csv(f'results/{algorithm}.csv', results)


if __name__ == '__main__':
    init_state = State.generate()

    print(f'Generating results for initial state:\n {init_state}')

    generate_algorithm_results("bfs", init_state)
    generate_algorithm_results("dfs", init_state)
    generate_algorithm_results("iddfs", init_state, limit=10)
    generate_algorithm_results("iddfs", init_state, limit=50)
    generate_algorithm_results("iddfs", init_state, limit=100)
    generate_algorithm_results("hgs", init_state, heuristic="manhattan")
    generate_algorithm_results("hgs", init_state, heuristic="hamming")
    generate_algorithm_results("hgs", init_state, heuristic="overestimated")
    generate_algorithm_results("hls", init_state, heuristic="manhattan")
    generate_algorithm_results("hls", init_state, heuristic="hamming")
    generate_algorithm_results("hls", init_state, heuristic="overestimated")
    generate_algorithm_results("a_star", init_state, heuristic="manhattan")
    generate_algorithm_results("a_star", init_state, heuristic="hamming")
    generate_algorithm_results("a_star", init_state, heuristic="overestimated")
