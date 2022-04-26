from numpy import tanh, exp

NON_LINEAR_FUNCTIONS = {
    "tanh": (
        lambda b, h: tanh(b * h),
        lambda b, h: b * (1 - tanh(b * h) ** 2)
    ),
    "logistic": (
        lambda b, h: 1 / (1 + exp(-2 * b * h)),
        lambda b, h: 2 * b * (1 / (1 + exp(-2 * b * h))) * (1 - (1 / (1 + exp(-2 * b * h))))
    )
}

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_THRESHOLD = 5000


class NeuronConfig:

    def __init__(self,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 threshold: int = DEFAULT_THRESHOLD,
                 normalize: bool = False,
                 g: str = None,
                 b: float = 0.01
                 ):
        self.normalize = normalize
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.b = b
        if g:
            self.g = NON_LINEAR_FUNCTIONS[g]
