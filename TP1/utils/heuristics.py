from utils.board import State, BOARD_SIZE, Position, OBJECTIVE_STATE

OBJECTIVE_POSITIONS = {
    1: Position(0, 0),
    2: Position(0, 1),
    3: Position(0, 2),
    4: Position(1, 0),
    5: Position(1, 1),
    6: Position(1, 2),
    7: Position(2, 0),
    8: Position(2, 1),
    9: Position(2, 2)
}


def manhattan_distance(actual_state: State):
    distance = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            actual = actual_state.state[i][j]
            if actual != 0:
                distance += abs(i - OBJECTIVE_POSITIONS[actual].i) + abs(j - OBJECTIVE_POSITIONS[actual].j)
    return distance


def overestimated_manhattan_distance(actual_state: State):
    distance = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            actual = actual_state.state[i][j]
            distance += abs(i - OBJECTIVE_POSITIONS[actual].i) + abs(j - OBJECTIVE_POSITIONS[actual].j)
    return distance


def hamming_distance(actual_state: State):
    n_tiles = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            actual = actual_state.state[i][j]
            if actual != 0:
                if actual != OBJECTIVE_STATE.state[i][j]:
                    n_tiles += 1
    return n_tiles
