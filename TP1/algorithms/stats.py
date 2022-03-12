class Stats:
    def __init__(self):
        self.objective_found = False
        self.objective_distance = 0
        self.objective_cost = 0
        self.explored_nodes_count = 0
        self.border_nodes_count = 0
        self.start_time = 0
        self.end_time = 0

    def get_processing_time(self):
        return self.end_time - self.start_time

    def __str__(self):
        return f'Found: {self.objective_found}\nObjective Distance: {self.objective_distance}\n' \
               f'Objective Cost: {self.objective_cost}\nExplores Nodes: {self.explored_nodes_count}\n' \
               f'Border Nodes: {self.border_nodes_count}\nProcessing Time: {self.get_processing_time()}s\n'
