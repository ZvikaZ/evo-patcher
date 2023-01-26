import re
from pathlib import Path

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

logger = logging.getLogger(__name__)


def del_some_images(sender, data_dict):
    try:
        ids_to_keep = [str(sender.best_of_gen.id), str(sender.worst_of_gen.id)]
    except AttributeError:
        # TODO this is actually a bug on EcKity side, it doesn't assign these values of terminated immediately after first gen
        logger.debug("'sender' doesn't have best_of_gen, or worst_of_gen ; not deleting any image")
        return
    r = re.compile(r'.*__gen_.*_ind_(.*)__.*')
    p = Path('runs') / 'patches' / f'gen_{sender.generation_num}'
    for png in p.glob('*.png'):
        if r.match(png.stem).group(1) not in ids_to_keep:
            png.unlink()


def evolve(individuals, creation_max_depth, population_size, num_of_evolve_threads, num_of_images_threads,
           max_generation, random_seed, patch_ratio_x, patch_ratio_y, elitism_rate, bloat_weight,
           imagenet_path, batch_size, num_of_images, classes, threshold_size_ratio, threshold_confidence,
           fail_weight, prob_weight, abs_prob):
    function_set = [t_add, t_mul, t_sub, t_div, t_iflte, t_sin, t_cos, t_atan2, t_hypot, t_sigmoid]
    terminal_set = ['x', 'y']

    maximization_problem = True

    # first image is using RampedHalfAndHalfCreator ; after that, each image uses previous image's last population
    if individuals:
        logger.debug("Using previous individuals")
        creators = None
    else:
        logger.debug("Creating new individuals")
        creators = RampedHalfAndHalfCreator(init_depth=(2, creation_max_depth),
                                            terminal_set=terminal_set,
                                            function_set=function_set,
                                            erc_range=(-1, 1),
                                            bloat_weight=bloat_weight)

    # Initialize the evolutionary algorithm
    # TODO maximal growth factor of 4.0 (from FINCH) or max tree depth of 17 (Koza)
    algo = SimpleEvolution(
        Subpopulation(creators=creators,
                      individuals=individuals,
                      population_size=population_size,
                      evaluator=Evaluator(num_of_images_threads, imagenet_path, batch_size, num_of_images, classes,
                                          random_seed, patch_ratio_x, patch_ratio_y,
                                          threshold_size_ratio, threshold_confidence,
                                          fail_weight, prob_weight, abs_prob),
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

    # don't keep all dumped images - only best and worst fitness (except from the initial population)
    # algo.register('after_generation', del_some_images)

    # evolve the generated initial population
    algo.evolve()

    return algo


if __name__ == '__main__':
    evolve(creation_max_depth=6, population_size=200, max_generation=200, random_seed=1)
