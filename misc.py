import argparse
import os
import shutil
import re
import logging
from pathlib import Path

import torch

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


def initial_dump_images(imgs):
    p = Path('runs') / Path('initial')
    p.mkdir(parents=True, exist_ok=True)
    for img_d in imgs:
        img = img_d['img']
        shutil.copy(img, p / (Path(img).stem + "__" + img_d['label'] + Path(img).suffix))


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


def create_run_dir(all_runs_dir, run_name):
    # helper function for run.sh

    p = Path(all_runs_dir)
    p.mkdir(exist_ok=True)

    current_max = 0
    for run_dir in p.glob('run_*'):
        run_num = int(re.match('run_(\d+).*', run_dir.stem).group(1))
        if run_num > current_max:
            current_max = run_num
    if run_name:
        run_name = '_' + run_name
    run_p = p / (f'run_{current_max + 1}{run_name}')
    run_p.mkdir()

    print(run_p)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         description="Internal script, shouldn't be run by users")
    arg_parser.add_argument("--create-run-dir-all-runs", help=argparse.SUPPRESS)
    arg_parser.add_argument("--create-run-dir-run-name", help=argparse.SUPPRESS, default='')
    args = arg_parser.parse_args()

    if args.create_run_dir_all_runs:
        create_run_dir(args.create_run_dir_all_runs, args.create_run_dir_run_name)
