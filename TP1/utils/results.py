import csv
from statistics import mean, fmean
from typing import Iterable, List

from algorithms.stats import Stats
from config import Config
from utils.board import OBJECTIVE_STATE, State
from utils.node import Node
import matplotlib.pyplot as plt

import numpy as np


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


def generate_algorithm_results(algorithm: str, init_state: State,
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


def generate_all_csv():
    init_state = State(
        np.array(
            [
                [7, 3, 0],
                [1, 5, 2],
                [8, 4, 6]
            ],
            int
        )
    )

    print(f'Generating results for initial state:\n {init_state}')

    # generate_algorithm_results("bfs", init_state)
    # generate_algorithm_results("dfs", init_state)
    generate_algorithm_results("iddfs", init_state, limit=1)
    generate_algorithm_results("iddfs", init_state, limit=5)
    generate_algorithm_results("iddfs", init_state, limit=10)
    generate_algorithm_results("iddfs", init_state, limit=100)
    generate_algorithm_results("iddfs", init_state, limit=1000)
    generate_algorithm_results("iddfs", init_state, limit=10000)
    # generate_algorithm_results("hgs", init_state, heuristic="manhattan")
    # generate_algorithm_results("hgs", init_state, heuristic="hamming")
    # generate_algorithm_results("hgs", init_state, heuristic="overestimated")
    # generate_algorithm_results("hls", init_state, heuristic="manhattan")
    # generate_algorithm_results("hls", init_state, heuristic="hamming")
    # generate_algorithm_results("hls", init_state, heuristic="overestimated")
    # generate_algorithm_results("a_star", init_state, heuristic="manhattan")
    # generate_algorithm_results("a_star", init_state, heuristic="hamming")
    # generate_algorithm_results("a_star", init_state, heuristic="overestimated")

def _plot_uninformed_algorithms():
    algorithms = ["bfs", "dfs", "iddfs"]

    results = {
        'bfs': {
            'time': [],
            'cost': [],
        },
        'dfs': {
            'time': [],
            'cost': []
        },
        'iddfs': {
            'time': [],
            'cost': []
        }
    }

    for a in algorithms:
        with open(f'results/{a}.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                results[row['algorithm']]['time'].append(float(row['processing_time']))
                results[row['algorithm']]['cost'].append(int(row['objective_cost']))

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    IT = [12, 30, 1, 8, 22]
    ECE = [28, 6, 16, 5, 10]
    CSE = [29, 3, 24, 25, 17]

    # Set position of bar on X axis
    br1 = np.arange(len(IT))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, IT, color='r', width=barWidth,
            edgecolor='grey', label='IT')
    plt.bar(br2, ECE, color='g', width=barWidth,
            edgecolor='grey', label='ECE')
    plt.bar(br3, CSE, color='b', width=barWidth,
            edgecolor='grey', label='CSE')

    # Adding Xticks
    plt.xlabel('Branch', fontweight='bold', fontsize=15)
    plt.ylabel('Students passed', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(IT))],
               ['2015', '2016', '2017', '2018', '2019'])

    fig, time = plt.subplots()

    time.set_xlabel('Algorithm')
    time.set_ylabel('Time (s)')

    time.bar(results.keys(),
             list(map(lambda algo: fmean(algo['time']), results.values())),
             color="red")

    cost = time.twinx()

    cost.set_ylabel('Cost')

    cost.bar(results.keys(),
             list(map(lambda algo: mean(algo['cost']), results.values())),
             color="blue")

    plt.title('Time and Cost Uninformed Search')
    plt.show()


def plot_results():
    _plot_uninformed_algorithms()


if __name__ == '__main__':
    generate_all_csv()
