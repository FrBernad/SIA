from utils.backpack import Backpack, Chromosome


def benefit_weight_ratio(bp: Backpack, c: Chromosome) -> float:
    weight = bp.calculate_weight(c)
    benefit = bp.calculate_benefit(c)
    if weight > bp.max_weight:
        return 0 if not weight else round(benefit / weight, 2)
    return benefit


FITNESS_FUNCTIONS = {
    "benefit_weight_ratio": benefit_weight_ratio,
}
