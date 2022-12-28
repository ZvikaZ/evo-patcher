"""
This module implements the fitness class, which delivers the fitness function.
You will need to implement such a class to work with your own problem and fitness function.
"""
import shutil
import logging
from pathlib import Path
import torch
import torchvision.io
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

from explore import prepare, infer_images
from misc import get_scratch_dir

logger = logging.getLogger(__name__)


class Evaluator(SimpleIndividualEvaluator):
    """
    Compute the fitness of an individual.
    """

    def __init__(self):
        super().__init__()
        data = prepare()
        self.device = data['device']
        self.resnext = data['resnext']
        self.imagenet_data = data['imagenet_data']
        self.image_results = data['image_results']

        self.num_of_images_to_dump = 2
        # TODO move somewhere else?
        self.ratio_x = 0.3
        self.ratio_y = 0.3

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

        for i, img_result in enumerate(self.image_results):
            label = img_result['label']
            (scratch_dir / label).mkdir(exist_ok=True, parents=True)
            img_name = self.apply_patches(individual, img_result['img'], img_result['bb'], scratch_dir / label)
            self.dump_images(i, img_name)

        # TODO insert to fitness decreasing the inference probability?
        fitness = infer_images(scratch_dir, self.resnext, self.imagenet_data)
        shutil.rmtree(scratch_dir)

        self.dump_ind(individual, fitness)

        logger.info(f'{self.get_gen_id(individual)} : fitness is {fitness}')
        return fitness

    def apply_patches(self, individual, img, xyxy, label_dir):
        im = torchvision.io.read_image(img)  # TODO pass the image tensor around, instead of reading and writing
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
        torchvision.io.write_png(im, img_name)
        return img_name

    def get_patch(self, individual, width_x, width_y):
        patch = []
        # TODO vectorize
        for y in range(int(width_y)):
            line = []
            for x in range(int(width_x)):
                if individual.execute(x=x, y=y) > 0:
                    line.append(255)
                else:
                    line.append(0)
            patch.append(line)

        return torch.tensor(patch)

    def get_gen_id(self, individual):
        return f'gen_{individual.gen}_ind_{individual.id}'

    def dump_images(self, i, img_name):
        if i < self.num_of_images_to_dump:
            p = Path('runs') / 'dump' / 'patches'
            p.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_name, p)

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
