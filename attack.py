import glob
import logging
import os
import shutil
from pathlib import Path

from evolution import evolve

logger = logging.getLogger(__name__)


def clean_clone(ind):
    result = ind.clone()
    result.gen = 0
    result.cloned_from = []
    result.selected_by = []
    result.applied_operators = []
    try:
        result.set_fitness_not_evaluated()
    except ValueError:
        pass
    return result


def directories_mingling(i: int, name: str = 'single_attacks'):
    orig_dir = os.getcwd()
    p = Path(name) / f'attack_{i}'
    p.mkdir(parents=True)
    for f in glob.glob('persist*'):
        shutil.copy(f, p)
    os.chdir(p)
    return orig_dir


def single_images_attack(creation_max_depth, population_size, num_of_evolve_threads, num_of_images_threads,
                         max_generation, random_seed, patch_ratio_x, patch_ratio_y, elitism_rate, bloat_weight,
                         imagenet_path, batch_size, num_of_images, classes, threshold_size_ratio, threshold_confidence,
                         fail_weight, prob_weight, abs_prob, colors):
    classes = []  # ignore user's request, use all classes
    individuals = None
    shutil.rmtree('single_attacks', ignore_errors=True)
    for i in range(num_of_images):
        logger.info('*************************************')
        logger.info(f'Starting single-image attack #{i}')
        orig_dir = directories_mingling(i)
        algo = evolve(individuals=individuals,
                      creation_max_depth=creation_max_depth,
                      population_size=population_size,
                      num_of_evolve_threads=num_of_evolve_threads,
                      num_of_images_threads=num_of_images_threads,
                      max_generation=max_generation,
                      random_seed=random_seed + i,
                      patch_ratio_x=patch_ratio_x,
                      patch_ratio_y=patch_ratio_y,
                      elitism_rate=elitism_rate,
                      bloat_weight=bloat_weight,
                      imagenet_path=imagenet_path,
                      batch_size=batch_size,
                      num_of_images=1,
                      classes=classes,
                      threshold_size_ratio=threshold_size_ratio,
                      threshold_confidence=threshold_confidence,
                      fail_weight=fail_weight,
                      prob_weight=prob_weight,
                      abs_prob=abs_prob,
                      colors=colors, )
        if algo.best_of_run_.get_pure_fitness() > 0.8:
            # something more sophisticated?
            logger.debug('Saving individuals for next image')
            individuals = [clean_clone(ind) for ind in algo.population.sub_populations[0].individuals]
        else:
            logger.debug('Cleaning individuals for next image')
            if individuals is not None:
                individuals = [clean_clone(ind) for ind in individuals]
        os.chdir(orig_dir)


def attack(single_image, creation_max_depth, population_size, num_of_evolve_threads, num_of_images_threads,
           max_generation, random_seed, patch_ratio_x, patch_ratio_y, elitism_rate, bloat_weight,
           imagenet_path, batch_size, num_of_images, classes, threshold_size_ratio, threshold_confidence,
           fail_weight, prob_weight, abs_prob, colors):
    if single_image:
        # perform a Single-Image attack (on possibly many images)
        single_images_attack(creation_max_depth=creation_max_depth,
                             population_size=population_size,
                             num_of_evolve_threads=num_of_evolve_threads,
                             num_of_images_threads=num_of_images_threads,
                             max_generation=max_generation,
                             random_seed=random_seed,
                             patch_ratio_x=patch_ratio_x,
                             patch_ratio_y=patch_ratio_y,
                             elitism_rate=elitism_rate,
                             bloat_weight=bloat_weight,
                             imagenet_path=imagenet_path,
                             batch_size=batch_size,
                             num_of_images=num_of_images,
                             classes=classes,
                             threshold_size_ratio=threshold_size_ratio,
                             threshold_confidence=threshold_confidence,
                             fail_weight=fail_weight,
                             prob_weight=prob_weight,
                             abs_prob=abs_prob,
                             colors=colors)
    else:
        # perform a Universal Attack
        evolve(individuals=None,
               creation_max_depth=creation_max_depth,
               population_size=population_size,
               num_of_evolve_threads=num_of_evolve_threads,
               num_of_images_threads=num_of_images_threads,
               max_generation=max_generation,
               random_seed=random_seed,
               patch_ratio_x=patch_ratio_x,
               patch_ratio_y=patch_ratio_y,
               elitism_rate=elitism_rate,
               bloat_weight=bloat_weight,
               imagenet_path=imagenet_path,
               batch_size=batch_size,
               num_of_images=num_of_images,
               classes=classes,
               threshold_size_ratio=threshold_size_ratio,
               threshold_confidence=threshold_confidence,
               fail_weight=fail_weight,
               prob_weight=prob_weight,
               abs_prob=abs_prob,
               colors=colors)
