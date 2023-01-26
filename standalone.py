import sys
import importlib
from inspect import getmembers, isfunction
from pathlib import Path

import models_wrapper
from misc import get_device
from image_utils import apply_patches, get_dominant_color



def standalone(indfile, imagefile, ratio_x, ratio_y):
    device = get_device()
    sys.path.append(str(Path(indfile).parent))
    module = importlib.import_module(Path(indfile).stem)
    functions = [func for name, func in getmembers(module, isfunction) if name.startswith('func_')]
    assert len(functions) == 1
    func = functions[0]

    # patch = get_patch(func, 20, 20, (100, 150, 200), device)
    # print(patch)
    # import numpy as np
    # temp = np.moveaxis(patch.to('cpu').numpy(), 0, -1)
    # torchvision.io.write_png(patch.to('cpu'), 'patch.png')

    yolo = models_wrapper.YoloModel(device)
    yolo_results = yolo.infer([imagefile])
    assert len(yolo_results) == 1

    label_dir = Path('.')  # TODO?
    out_img = apply_patches(func, imagefile, yolo_results[0]['xyxy'], label_dir, ratio_x, ratio_y, 'patched', device)
    print(out_img)


if __name__ == '__main__':
    #TODO argparse
    standalone("runs/population/gen_0/gen_0_ind_1.py",
               "runs/initial/n04285008_13702__sports car.JPEG",
               0.4, 0.4)