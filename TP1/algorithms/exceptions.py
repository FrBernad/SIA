class InvalidConfigLimitException(Exception):

    def __init__(self):
        self.message = "Invalid limit for algorithm. Must be a positive integer bigger than 0."

    def __str__(self):
        return self.message


class InvalidAlgorithmException(Exception):
    def __init__(self):
        self.message = "Invalid algorithm specified. Select a valid one and try again."

    def __str__(self):
        return self.message


class MissingHeuristicException(Exception):
    def __init__(self):
        self.message = "Missing heuristic for informed search. Specify one and try again."

    def __str__(self):
        return self.message
