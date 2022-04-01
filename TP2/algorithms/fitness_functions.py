from utils.chromosome_factory import Chromosome
from utils.knapsack import Knapsack


def benefit_weight_ratio(knapsack: Knapsack, c: Chromosome) -> float:
    weight = c.weight
    benefit = c.benefit
    if weight > knapsack.max_weight:
        return benefit / weight
    return benefit


FITNESS_FUNCTIONS = {
    "benefit_weight_ratio": benefit_weight_ratio,
}
