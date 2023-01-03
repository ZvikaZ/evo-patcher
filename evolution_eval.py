"""
This module implements the fitness class, which delivers the fitness function.
"""
import shutil
import logging
from pathlib import Path

import torch
import torchvision.io
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

from image_utils import prepare, infer_images
from misc import get_scratch_dir

logger = logging.getLogger(__name__)


class Evaluator(SimpleIndividualEvaluator):
    """
    Compute the fitness of an individual.
    """

    def __init__(self, num_of_images_threads, imagenet_path, batch_size, num_of_images,
                 threshold_size_ratio, threshold_confidence):
        super().__init__()
        self.batch_size = batch_size
        self.num_of_images_threads = num_of_images_threads

        data = prepare(num_of_images_threads, imagenet_path, batch_size, num_of_images, threshold_size_ratio,
                       threshold_confidence)
        self.device = data['device']
        self.resnext = data['resnext']
        self.imagenet_data = data['imagenet_data']
        self.image_results = data['image_results']
        self.image_probs = data['image_probs']

        # TODO move to main.py
        self.num_of_images_to_dump = 2
        self.ratio_x = 0.4
        self.ratio_y = 0.4

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
        individual.execute_cache = {}
        scratch_dir = get_scratch_dir() / self.get_gen_id(individual)
        img_names = []

        for i, img_result in enumerate(self.image_results):
            label = img_result['label']
            (scratch_dir / label).mkdir(exist_ok=True, parents=True)
            img_names.append(self.apply_patches(individual, img_result['img'], img_result['bb'], scratch_dir / label))

        del individual.execute_cache

        y, y_hat, probs = infer_images(scratch_dir, self.resnext, self.imagenet_data, self.batch_size,
                                       self.num_of_images_threads)

        model_fail_rate = (y != y_hat).count_nonzero() / len(y)
        avg_prob_diff = (self.image_probs - probs).mean()

        fitness = model_fail_rate * 0.7 + avg_prob_diff * 0.3
        fitness = fitness.item()

        for i, img_name in enumerate(img_names):
            self.dump_images(i, individual.gen, img_name, fitness)
        shutil.rmtree(scratch_dir)

        self.dump_ind(individual, fitness)

        logger.info(f'{self.get_gen_id(individual)} : fitness is {fitness:.4f}')
        return fitness

    def apply_patches(self, individual, img, xyxy, label_dir):
        im = torchvision.io.read_image(img).to(self.device)
        for x1, y1, x2, y2, confidence, label in xyxy:
            width_x = int(x2 - x1)
            width_y = int(y2 - y1)
            patch_width_x = int(width_x * self.ratio_x)
            patch_width_y = int(width_y * self.ratio_y)
            start_x = int(x1 + (width_x - patch_width_x) / 2)
            start_y = int(y1 + (width_y - patch_width_y) / 2)
            patch = self.get_patch(individual, patch_width_x, patch_width_y)
            im[:, start_y:start_y + patch_width_y, start_x:start_x + patch_width_x] = patch
        img_name = (label_dir / f'{Path(img).stem}__{self.get_gen_id(individual)}.png').as_posix()
        torchvision.io.write_png(im.to('cpu'), img_name)
        return img_name

    def get_patch(self, individual, width_x, width_y):
        yy, xx = torch.meshgrid(torch.arange(width_y), torch.arange(width_x))
        xx = xx.to(self.device)
        yy = yy.to(self.device)
        result = self.execute(individual, x=xx, y=yy)
        try:
            return (result > 0).int() * 255
        except AttributeError:
            assert type(result) == float
            if result > 0:
                return torch.ones_like(xx)
            else:
                return torch.zeros_like(xx)

    def execute(self, individual, x, y):
        key = (individual, x, y)
        if key not in individual.execute_cache:
            individual.execute_cache[key] = individual.execute(x=x, y=y)
        else:
            logger.info("USING CACHE")  # TODO if it's never printed, delete this mechanism
        return individual.execute_cache[key]

    def get_gen_id(self, individual):
        return f'gen_{individual.gen}_ind_{individual.id}'

    def dump_images(self, i, gen, img_name, fitness):
        if i < self.num_of_images_to_dump:
            p = Path('runs') / 'dump' / 'patches' / f'gen_{gen}'
            p.mkdir(parents=True, exist_ok=True)
            orig = Path(img_name)
            shutil.copy(img_name, p / f'{orig.stem}__fitness_{fitness:.3f}{orig.suffix}')

    def dump_ind(self, individual, fitness):
        p = Path('runs') / 'dump' / 'population' / f'gen_{individual.gen}'
        p.mkdir(parents=True, exist_ok=True)
        with open(p / (self.get_gen_id(individual) + '.log'), 'w') as f:
            f.write(f'fitness: {fitness}\n')
            f.write(f'gen: {individual.gen} , id: {individual.id}\n')
            f.write(f'cloned from: {individual.cloned_from}\n')
            f.write(f'tree size: {individual.size()}\n')
            f.write(f'tree depth: {individual.depth()}\n')
            f.write('code:\n')
            f.write(str(individual))
            f.write('\n')
