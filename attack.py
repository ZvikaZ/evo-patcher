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
    return result


def directories_mingling(i):
    orig_dir = os.getcwd()
    p = Path('single_attacks') / f'single_attack_{i}'
    p.mkdir(parents=True)
    for f in glob.glob('persist*'):
        shutil.copy(f, p)
    os.chdir(p)
    return orig_dir


def single_images_attack(creation_max_depth, population_size, num_of_evolve_threads, num_of_images_threads,
                         max_generation, random_seed, patch_ratio_x, patch_ratio_y, elitism_rate, bloat_weight,
                         imagenet_path, batch_size, num_of_images, classes, threshold_size_ratio, threshold_confidence):
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
                      classes=[],  # ignore user's request, use all classes
                      threshold_size_ratio=threshold_size_ratio,
                      threshold_confidence=threshold_confidence)
        if algo.best_of_run_.get_pure_fitness() > 0.8:
            # something more sophisticated?
            logger.debug('Saving individuals for next image')
            individuals = [clean_clone(ind) for ind in algo.population.sub_populations[0].individuals]
        os.chdir(orig_dir)


def attack(single_image, creation_max_depth, population_size, num_of_evolve_threads, num_of_images_threads,
           max_generation, random_seed, patch_ratio_x, patch_ratio_y, elitism_rate, bloat_weight,
           imagenet_path, batch_size, num_of_images, classes, threshold_size_ratio, threshold_confidence):
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
                             threshold_confidence=threshold_confidence)
    else:
        # perform a Universal Attack
        evolve(creation_max_depth=creation_max_depth,
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
               threshold_confidence=threshold_confidence)
