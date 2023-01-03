from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.ramped_hh import RampedHalfAndHalfCreator
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.erc_mutation import ERCMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_avg_worst_size_tree_statistics import BestAverageWorstSizeTreeStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker

from evolution_eval import Evaluator
from evolution_func import *


def evolve(creation_max_depth, population_size, num_of_evolve_threads, num_of_images_threads, max_generation,
           random_seed, patch_ratio_x, patch_ratio_y, elitism_rate,
           imagenet_path, batch_size, num_of_images, threshold_size_ratio, threshold_confidence):
    function_set = [t_add, t_mul, t_sub, t_div, t_iflte, t_sin, t_cos, t_atan2, t_hypot]
    terminal_set = ['x', 'y']

    maximization_problem = True

    # Initialize the evolutionary algorithm
    # TODO maximal growth factor of 4.0 (from FINCH) or max tree depth of 17 (Koza)
    algo = SimpleEvolution(
        Subpopulation(creators=RampedHalfAndHalfCreator(init_depth=(2, creation_max_depth),
                                                        terminal_set=terminal_set,
                                                        function_set=function_set,
                                                        erc_range=(-1, 1),
                                                        bloat_weight=0.0001),  # TODO is it enough?
                      population_size=population_size,
                      evaluator=Evaluator(num_of_images_threads, imagenet_path, batch_size, num_of_images,
                                          patch_ratio_x, patch_ratio_y, threshold_size_ratio, threshold_confidence),
                      higher_is_better=maximization_problem,
                      elitism_rate=elitism_rate,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          SubtreeCrossover(probability=0.8, arity=2),
                          SubtreeMutation(probability=0.2, arity=1),
                          ERCMutation(probability=0.05, arity=1)  # TODO is ERC working??
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=7, higher_is_better=maximization_problem), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=num_of_evolve_threads,
        max_generation=max_generation,
        random_seed=random_seed,  # TODO allow free seeding
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=1, threshold=0.001),
        statistics=BestAverageWorstSizeTreeStatistics(
            format_string='fitness: best {:.4f}, worst {:.4f}, average {:.4f}. average size {}')
    )

    # evolve the generated initial population
    algo.evolve()


if __name__ == '__main__':
    evolve(creation_max_depth=6, population_size=200, max_generation=200, random_seed=1)
