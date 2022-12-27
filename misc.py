import shutil
from pathlib import Path
import torch


def get_device():
    if torch.cuda.is_available():
        print('Using CUDA')
        return 'cuda'
    else:
        print('Using CPU')
        return 'cpu'


def dump_images(imgs, name=None):
    p = Path('runs') / Path('dump')
    if name == "initial":
        shutil.rmtree(p, ignore_errors=True)
    if name:
        p = p / name
    p.mkdir(parents=True, exist_ok=True)
    for img, label in imgs:
        label_p = p / label
        label_p.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, label_p / (Path(img).stem + "__" + label + Path(img).suffix))
