from utils.backpack import Backpack, Chromosome


def benefit_weight_ratio(bp: Backpack, c: Chromosome) -> float:
    weight = bp.calculate_weight(c)
    return 0 if not weight else round(bp.calculate_benefits(c) / (bp.calculate_weight(c)), 2)


def no_overweight_fitness(bp: Backpack, c: Chromosome) -> int:
    return 0 if bp.calculate_weight(c) > bp.max_weight else bp.calculate_benefits(c)


FITNESS_FUNCTIONS = {
    "benefit_weight_ratio": benefit_weight_ratio,
    "no_overweight_fitness": no_overweight_fitness
}
