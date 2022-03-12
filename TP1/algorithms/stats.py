class Stats:
    def __init__(self):
        self.explored_nodes_count = 0
        self.objective_distance = 0
        self.start_time = 0
        self.end_time = 0

    def get_processing_time(self):
        return self.end_time - self.start_time
