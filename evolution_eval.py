"""
This module implements the fitness class, which delivers the fitness function.
You will need to implement such a class to work with your own problem and fitness function.
"""
import shutil
from pathlib import Path
import torch
import torchvision.io
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

from explore import prepare, infer_images


class Evaluator(SimpleIndividualEvaluator):
    """
    Compute the fitness of an individual.
    """

    def __init__(self):
        super().__init__()
        self.data = prepare()

        # TODO move somewhere else?
        self.ratio_x = 0.3
        self.ratio_y = 0.3

    def get_temp_dir(self, id, img_label):
        # TODO use SSD, as described at https://www.ise.bgu.ac.il/clusters/ISE_CS_DT_GpuClusterUserGuide.pdf
        p = Path('runs') / 'dump' / id / img_label
        p.mkdir(exist_ok=True, parents=True)
        return p

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
        bb = self.data['bounding_boxes']
        for i, (img, label) in enumerate(self.data['imgs_with_labels']):
            assert Path(img).stem == Path(bb.files[i]).stem
            self.apply_patches(individual, img, label, bb.xyxy[i])

        images_dir = self.get_temp_dir(f'gen_{individual.gen}_ind_{individual.id}', '')  # TODO improve
        fitness = infer_images(images_dir, self.data['resnext'], self.data['imagenet_data'])
        print(f'gen_{individual.gen}_ind_{individual.id} : fitness is {fitness}')
        return fitness

    def apply_patches(self, individual, img, img_label, xyxy):
        p = Path('runs') / 'dump' / 'patches'
        p.mkdir(parents=True, exist_ok=True)

        im = torchvision.io.read_image(img)  # TODO pass the image tensor around, instead of reading and writing
        for x1, y1, x2, y2, prob, label in xyxy:
            width_x = int(x2 - x1)
            width_y = int(y2 - y1)
            patch_width_x = int(width_x * self.ratio_x)
            patch_width_y = int(width_y * self.ratio_y)
            start_x = int(x1 + (width_x - patch_width_x) / 2)
            start_y = int(y1 + (width_y - patch_width_y) / 2)
            patch = self.get_patch(individual, patch_width_x, patch_width_y)
            im[:, start_y:start_y + patch_width_y, start_x:start_x + patch_width_x] = patch
        img_name = (p / f'{Path(img).stem}__gen_{individual.gen}_ind_{individual.id}.png').as_posix()
        torchvision.io.write_png(im, img_name)

        # TODO write to dir from get_temp_dir ; and optionally also dump to dump/patches
        temp_p = self.get_temp_dir(f'gen_{individual.gen}_ind_{individual.id}', img_label)
        shutil.copy(img_name, temp_p)

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
