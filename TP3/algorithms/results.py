import numpy as np

np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:f}'.format})


class Results:

    def __init__(self, expected_output, predicted_output, total_time, total_epochs, min_error):
        self.min_error = min_error
        self.total_epochs = total_epochs
        self.total_time = total_time
        self.predicted_output = predicted_output
        self.expected_output = expected_output

    def print(self, remove_ws=False):
        print("\t-Expected Output:")
        if remove_ws:
            print(f'{" ".join(self.expected_output.__str__().split())}')
        else:
            print(self.expected_output)

        print("\t-Predicted Output:")
        if remove_ws:
            print(f'{" ".join(self.predicted_output.__str__().split())}')
        else:
            print(self.predicted_output)
        print(f"\t-Total Time: {self.total_time}s")
        print(f"\t-Total Epochs: {self.total_epochs}")
        print(f"\t-Min Error: {self.min_error}")
