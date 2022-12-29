import os
import shutil
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        logger.info('Using CUDA')
        return 'cuda'
    else:
        logger.warning('Using CPU')
        return 'cpu'


def set_logger(logger_name, debug=True, logfile='run.log'):
    try:
        os.remove(logfile)
    except:
        pass
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level,
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(logfile)
                        ])
    # these are too noisy - disable them
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logging.getLogger(logger_name)


def dump_images(imgs, name=None):
    # TODO revise this
    p = Path('runs') / Path('dump')
    if name == "initial":
        shutil.rmtree('runs', ignore_errors=True)  # TODO move this to better place
    if name:
        p = p / name
    p.mkdir(parents=True, exist_ok=True)
    for img, label in imgs:
        label_p = p / label
        label_p.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, label_p / (Path(img).stem + "__" + label + Path(img).suffix))


def get_scratch_dir():
    p = Path('/scratch') / os.environ['USER'] / os.environ['SLURM_JOB_ID']
    if p.is_dir():
        p = p / f'pid_{os.getpid()}'
        p.mkdir(exist_ok=True)
        return p
    else:
        logger.info(
            'Using ./scratch dir ; you might want to override "get_scratch_dir()" to use some fast local directory')
        p = Path('scratch')
        p.mkdir(exist_ok=True)
        return p
