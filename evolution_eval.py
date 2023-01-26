"""
This module implements the fitness class, which delivers the fitness function.
"""
import shutil
import logging
from pathlib import Path

import torch
import torchvision.io
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

from image_utils import prepare, infer_images, apply_patches
from misc import get_scratch_dir

logger = logging.getLogger(__name__)


class Evaluator(SimpleIndividualEvaluator):
    """
    Compute the fitness of an individual.
    """

    def __init__(self, num_of_images_threads, imagenet_path, batch_size, num_of_images, classes, random_seed,
                 patch_ratio_x, patch_ratio_y, threshold_size_ratio, threshold_confidence,
                 fail_weight, prob_weight, abs_prob):
        super().__init__()
        self.batch_size = batch_size
        self.num_of_images_threads = num_of_images_threads
        self.ratio_x = patch_ratio_x
        self.ratio_y = patch_ratio_y

        data = prepare(num_of_images_threads, imagenet_path, batch_size, num_of_images, classes, random_seed,
                       threshold_size_ratio, threshold_confidence)
        self.device = data['device']
        self.resnext = data['resnext']
        self.imagenet_data = data['imagenet_data']
        self.image_results = data['image_results']
        self.image_probs = data['image_probs']

        self.num_of_images_to_dump = 8
        self.fail_weight = fail_weight
        self.prob_weight = prob_weight
        self.abs_prob = abs_prob
        logger.debug(f'fail_weight: {self.fail_weight}, prob_weight: {self.prob_weight}, abs_prob: {self.abs_prob}')

    def _evaluate_individual(self, individual):
        """
        Parameters
        ----------
        individual : Tree
            An individual program tree in the gp population, whose fitness needs to be computed.
            Makes use of GPTree.execute, which runs the program.
            Calling `gptree.execute` must use keyword arguments that match the terminal-set variables.
            For example, if the terminal set includes `x` and `y` then the call is `gptree.execute(x=..., y=...)`.

        Returns
        -------
        float
            fitness value
        """
        scratch_dir = get_scratch_dir() / self.get_gen_id(individual)
        img_names = []

        for i, img_result in enumerate(self.image_results):
            label = img_result['label']
            (scratch_dir / label).mkdir(exist_ok=True, parents=True)
            img_names.append(apply_patches(individual.execute, img_result['img'], img_result['bb'], scratch_dir / label,
                                           self.ratio_x, self.ratio_y, self.get_gen_id(individual), self.device))

        y, y_hat, probs = infer_images(scratch_dir, self.resnext, self.imagenet_data, self.batch_size,
                                       self.num_of_images_threads)

        model_fail_rate = (y != y_hat).count_nonzero() / len(y)
        if 'first' in self.abs_prob:  # used to be 'abs_prob=True'
            avg_prob_diff = 1 - probs.mean()
        elif 'second' in self.abs_prob:
            avg_prob_diff = 1 - (probs[y == y_hat]).mean()
        elif 'diff' in self.abs_prob:  # used to be 'abs_prob=False'
            avg_prob_diff = (self.image_probs[y == y_hat] - probs[y == y_hat]).mean()
        else:
            raise ValueError

        if 'prop' in self.abs_prob:
            temp = 1 - avg_prob_diff  # avg_prob_diff (first/second): 1 is worst , 0 is best ; temp: 1 is best , 0 is worst
            temp *= (
                    1 - model_fail_rate)  # model_fail_rate: 0 - all images need perturbations, 1 - no images needs; 1-mfr: rate that need perturbation
            avg_prob_diff = 1 - temp
        elif 'full' in self.abs_prob:
            pass
        else:
            raise ValueError

        if not avg_prob_diff.isnan():
            fitness = model_fail_rate * self.fail_weight + avg_prob_diff * self.prob_weight
        else:
            # all y are different from y_hat ; couldn't compute avg_prob_diff ;
            # it probably means that model_fail_rate is 1, and we won
            logger.info(f'avg_prob_diff is nan ; fitness == model_fail_rate == {model_fail_rate}')
            fitness = model_fail_rate
        fitness = fitness.item()

        # for i, img_name in enumerate(img_names):
        #     self.dump_images(i, individual.gen, img_name, fitness)
        shutil.rmtree(scratch_dir)

        self.dump_ind(individual, fitness, model_fail_rate, avg_prob_diff, y, y_hat, probs)

        logger.debug(
            f'{self.get_gen_id(individual)} : fitness is {fitness:.4f}, model_fail_rate={model_fail_rate:.4f}, avg_prob_diff={avg_prob_diff:.4f}')
        return fitness

    def get_gen_id(self, individual):
        return f'gen_{individual.gen}_ind_{individual.id}'

    def dump_images(self, i, gen, img_name, fitness):
        # note that most of these images will be later deleted by del_some_images(..)
        if gen == 0 or i < self.num_of_images_to_dump:
            p = Path('runs') / 'patches' / f'gen_{gen}'
            p.mkdir(parents=True, exist_ok=True)
            orig = Path(img_name)
            shutil.copy(img_name, p / f'{orig.stem}__fitness_{fitness:.3f}{orig.suffix}')

    def dump_ind(self, individual, fitness, model_fail_rate, avg_prob_diff, y, y_hat, probs):
        p = Path('runs') / 'population' / f'gen_{individual.gen}'
        p.mkdir(parents=True, exist_ok=True)
        with open(p / (self.get_gen_id(individual) + '.py'), 'w') as f:
            f.write('"""\n')
            f.write(f'gen: {individual.gen} , id: {individual.id}\n')
            f.write(f'fitness: {fitness}\n')
            f.write(f'    model_fail_rate: {model_fail_rate}    ,   avg_prob_diff: {avg_prob_diff}\n')
            f.write(f'    y     : {y}\n')
            f.write(f'    y_hat : {y_hat}\n')
            f.write(f'    probs (orig): {self.image_probs}\n')
            f.write(f'    probs (ind) : {probs}\n')
            f.write(f'cloned from       : {individual.cloned_from}\n')
            f.write(f'selected by       : {individual.selected_by}\n')
            f.write(f'applied operators : {individual.applied_operators}\n')
            f.write(f'tree size  : {individual.size()}\n')
            f.write(f'tree depth : {individual.depth()}\n')
            f.write('code:\n')
            f.write('"""\n\n')
            f.write('from evolution_func import *\n\n\n')
            f.write(individual.__str__(use_python_syntax=True))
            f.write('\n')
