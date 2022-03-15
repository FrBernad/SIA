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

    intermediate_nodes.pop()

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

    generate_algorithm_results("bfs", init_state)
    generate_algorithm_results("dfs", init_state)
    generate_algorithm_results("iddfs", init_state, limit=1)
    generate_algorithm_results("iddfs", init_state, limit=10)
    generate_algorithm_results("iddfs", init_state, limit=100)
    generate_algorithm_results("iddfs", init_state, limit=1000)
    generate_algorithm_results("iddfs", init_state, limit=10000)
    generate_algorithm_results("iddfs", init_state, limit=50000)
    generate_algorithm_results("hgs", init_state, heuristic="manhattan")
    generate_algorithm_results("hgs", init_state, heuristic="hamming")
    generate_algorithm_results("hgs", init_state, heuristic="overestimated")
    generate_algorithm_results("hls", init_state, heuristic="manhattan")
    generate_algorithm_results("hls", init_state, heuristic="hamming")
    generate_algorithm_results("hls", init_state, heuristic="overestimated")
    generate_algorithm_results("a_star", init_state, heuristic="manhattan")
    generate_algorithm_results("a_star", init_state, heuristic="hamming")
    generate_algorithm_results("a_star", init_state, heuristic="overestimated")


def _plot_uninformed_algorithms():
    algorithms = ["bfs", "dfs", "iddfs", "hgs", "hls", "a_star"]

    results = {
        'bfs': {
            'time': [],
            'cost': [],
        },
        'dfs': {
            'time': [],
            'cost': []
        },
        'iddfs_l1': {
            'time': [],
            'cost': []
        },
        'iddfs_l2': {
            'time': [],
            'cost': []
        },
        'iddfs_l3': {
            'time': [],
            'cost': []
        },
        'iddfs_l4': {
            'time': [],
            'cost': []
        }
        ,
        'iddfs_l5': {
            'time': [],
            'cost': []
        },
        'iddfs_l6': {
            'time': [],
            'cost': []
        },
        'manhattan': {
            'hgs': {
                'time': [],
                'cost': []
            },
            'hls': {
                'time': [],
                'cost': []
            },
            'a_star': {
                'time': [],
                'cost': []
            }
        },
        'hamming': {
            'hgs': {
                'time': [],
                'cost': []
            },
            'hls': {
                'time': [],
                'cost': []
            },
            'a_star': {
                'time': [],
                'cost': []
            }
        },
        'overestimated': {
            'hgs': {
                'time': [],
                'cost': []
            },
            'hls': {
                'time': [],
                'cost': []
            },
            'a_star': {
                'time': [],
                'cost': []
            }
        }
    }

    for a in algorithms:
        with open(f'results/{a}.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if a == 'iddfs':
                    if row['limit'] == '1':
                        results['iddfs_l1']['time'].append(float(row['processing_time']))
                        results['iddfs_l1']['cost'].append(int(row['objective_cost']))
                    if row['limit'] == '10':
                        results['iddfs_l2']['time'].append(float(row['processing_time']))
                        results['iddfs_l2']['cost'].append(int(row['objective_cost']))
                    if row['limit'] == '100':
                        results['iddfs_l3']['time'].append(float(row['processing_time']))
                        results['iddfs_l3']['cost'].append(int(row['objective_cost']))
                    if row['limit'] == '1000':
                        results['iddfs_l4']['time'].append(float(row['processing_time']))
                        results['iddfs_l4']['cost'].append(int(row['objective_cost']))
                    if row['limit'] == '10000':
                        results['iddfs_l5']['time'].append(float(row['processing_time']))
                        results['iddfs_l5']['cost'].append(int(row['objective_cost']))
                    if row['limit'] == '50000':
                        results['iddfs_l6']['time'].append(float(row['processing_time']))
                        results['iddfs_l6']['cost'].append(int(row['objective_cost']))
                elif h := row.get('heuristic'):
                    results[h][row['algorithm']]['time'].append(float(row['processing_time']))
                    results[h][row['algorithm']]['cost'].append(int(row['objective_cost']))
                else:
                    results[row['algorithm']]['time'].append(float(row['processing_time']))
                    results[row['algorithm']]['cost'].append(int(row['objective_cost']))

    # ### TIEMPO DE BFS, DFS Y TRES LIMITES DE IDDFS ###
    #
    # fig, time = plt.subplots(figsize=(10, 7))
    #
    # time.set_xlabel('Algorithm')
    # time.set_ylabel('Time (s)')
    #
    # algorithms = ["bfs", "dfs", "iddfs\n(limit 1)", "iddfs\n(limit 100)", "iddfs\n(limit 10000)",
    #               "iddfs\n(limit 50000)"]
    #
    # time_list = [results['bfs']['time'], results['dfs']['time'], results['iddfs_l1']['time'],
    #              results['iddfs_l3']['time'], results['iddfs_l5']['time'], results['iddfs_l6']['time']]
    #
    # time.bar(list(map(lambda algo: algo.upper(), algorithms)),
    #          list(map(lambda time: fmean(time), time_list)),
    #          color="blue")
    #
    # plt.title("Tiempo Algoritmos Desinformados")
    # plt.show()
    #
    # ### COSTO DE BFS DFS Y TRES LIMITES IDDFS ###
    #
    # fig, cost = plt.subplots(figsize=(10, 7))
    #
    # cost.set_xlabel('Algorithm')
    # cost.set_ylabel('Cost')
    #
    # algorithms = ["bfs", "dfs", "iddfs\n(limit 1)", "iddfs\n(limit 100)", "iddfs\n(limit 10000)",
    #               "iddfs\n(limit 50000)"]
    #
    # cost_list = [results['bfs']['cost'], results['dfs']['cost'], results['iddfs_l1']['cost'],
    #              results['iddfs_l3']['cost'], results['iddfs_l5']['cost'], results['iddfs_l6']['cost']]
    #
    # cost.bar(list(map(lambda algo: algo.upper(), algorithms)),
    #          list(map(lambda cost: mean(cost), cost_list)),
    #          color="red")
    #
    # plt.title("Costo Algoritmos Desinformados")
    # plt.show()
    #
    #
    # ### TIEMPO IDDFS CON DISTINTOS LIMITES ###
    #
    # fig, iddfs_time_graph = plt.subplots(figsize=(10, 7))
    #
    # limits = ['1', '10', '100', '1000', '10000', '50000']
    #
    # iddfs_time_graph.set_xlabel('Limit')
    # iddfs_time_graph.set_ylabel('Time (s)')
    #
    # iddfs_time_list = [results['iddfs_l1']['time'], results['iddfs_l2']['time'], results['iddfs_l3']['time'],
    #                    results['iddfs_l4']['time'], results['iddfs_l5']['time'], results['iddfs_l6']['time']]
    #
    # iddfs_time_graph.bar(limits,
    #                      list(map(lambda limit: fmean(limit), iddfs_time_list)),
    #                      color="blue")
    #
    # plt.title("Tiempo IDDFS Limites Variables")
    # plt.show()
    #
    # ### COSTO IDDFS CON DISTINTOS LIMITES ###
    #
    # fig, iddfs_cost_graph = plt.subplots(figsize=(10, 7))
    #
    # limits = ['1', '10', '100', '1000', '10000', '50000']
    #
    # iddfs_cost_graph.set_xlabel('Limit')
    # iddfs_cost_graph.set_ylabel('Cost')
    #
    # iddfs_cost_list = [results['iddfs_l1']['cost'], results['iddfs_l2']['cost'], results['iddfs_l3']['cost'],
    #                    results['iddfs_l4']['cost'], results['iddfs_l5']['cost'], results['iddfs_l6']['cost']]
    #
    # iddfs_cost_graph.bar(limits,
    #                      list(map(lambda limit: mean(limit), iddfs_cost_list)),
    #                      color="red")
    #
    # plt.title("Costo IDDFS Limites Variables")
    # plt.show()

    ### TIEMPO DE HLS, HGS Y A_STAR###

    fig, time = plt.subplots(figsize=(12, 7))

    time.set_xlabel('Algorithm')
    time.set_ylabel('Time (s)')

    algorithms = ["hls\n(manhattan)", "hls\n(hamming)", "hls\n(overestimated)",
                  "hgs\n(manhattan)", "hgs\n(hamming)", "hgs\n(overestimated)",
                  "A*\n(manhattan)", "A*\n(hamming)", "A*\n(overestimated)"]

    time_list = [results['manhattan']['hls']['time'], results['hamming']['hls']['time'],
                 results['overestimated']['hls']['time'],
                 results['manhattan']['hgs']['time'], results['hamming']['hgs']['time'],
                 results['overestimated']['hgs']['time'],
                 results['manhattan']['a_star']['time'], results['hamming']['a_star']['time'],
                 results['overestimated']['a_star']['time']]

    time.bar(list(map(lambda algo: algo.upper(), algorithms)),
             list(map(lambda time: fmean(time), time_list)),
             color="blue")

    plt.title("Tiempo Algoritmos Informados")
    plt.show()

    ### COSTO DE HLS, HGS Y A_STAR###

    fig, cost = plt.subplots(figsize=(12, 7))

    cost.set_xlabel('Algorithm')
    cost.set_ylabel('Cost')

    algorithms = ["hls\n(manhattan)", "hls\n(hamming)", "hls\n(overestimated)",
                  "hgs\n(manhattan)", "hgs\n(hamming)", "hgs\n(overestimated)",
                  "A*\n(manhattan)", "A*\n(hamming)", "A*\n(overestimated)"]

    cost_list = [results['manhattan']['hls']['cost'], results['hamming']['hls']['cost'],
                 results['overestimated']['hls']['cost'],
                 results['manhattan']['hgs']['cost'], results['hamming']['hgs']['cost'],
                 results['overestimated']['hgs']['cost'],
                 results['manhattan']['a_star']['cost'], results['hamming']['a_star']['cost'],
                 results['overestimated']['a_star']['cost']]

    cost.bar(list(map(lambda algo: algo.upper(), algorithms)),
             list(map(lambda limit: mean(limit), cost_list)),
             color="red")

    plt.title("Costo Algoritmos Informados")
    plt.show()


def plot_results():
    _plot_uninformed_algorithms()


if __name__ == '__main__':
    plot_results()
