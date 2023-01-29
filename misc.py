import argparse
import shutil
import re
import logging
from pathlib import Path
from typing import List, Dict, Union

import torch

logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        logger.info('Using CUDA')
        return 'cuda'
    else:
        logger.warning('Using CPU')
        return 'cpu'


def set_logger(logger_name: str, debug: bool = True, logfile: str = 'run.log') -> logging.Logger:
    Path(logfile).unlink(missing_ok=True)
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


def initial_dump_images(imgs: List[Dict[str, Union[str, torch.Tensor]]]) -> None:
    path = Path('runs') / Path('initial')
    path.mkdir(parents=True, exist_ok=True)
    for img_d in imgs:
        img = img_d['img']
        assert isinstance(img, str)
        assert isinstance(img_d['label'], str)
        shutil.copy(img, path / (Path(img).stem + "__" + img_d['label'] + Path(img).suffix))


def create_run_dir(all_runs_dir: str, run_name: str) -> None:
    # helper function for run.sh

    path = Path(all_runs_dir)
    path.mkdir(exist_ok=True)

    current_max = 0
    for run_dir in path.glob('run_*'):
        match = re.match(r'run_(\d+).*', run_dir.stem)
        assert match
        run_num = int(match.group(1))
        if run_num > current_max:
            current_max = run_num
    if run_name:
        run_name = '_' + run_name
    run_p = path / (f'run_{current_max + 1}{run_name}')
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
