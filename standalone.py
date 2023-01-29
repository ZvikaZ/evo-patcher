import sys
import importlib
from inspect import getmembers, isfunction
from pathlib import Path

import torchvision

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

    yolo = models_wrapper.YoloModel(device)
    yolo_results = yolo.infer([imagefile])
    assert len(yolo_results) == 1

    im = torchvision.io.read_image(imagefile).to(device)
    apply_patches(func, im, yolo_results[0]['xyxy'], ratio_x, ratio_y, device)
	
    out_img = imagefile + "_patched.png"  # TODO
    torchvision.io.write_png(im.to('cpu'), out_img)
    print(out_img)


if __name__ == '__main__':
    # TODO argparse
    standalone("runs/population/gen_0/gen_0_ind_1.py",
               "runs/initial/n04285008_13702__sports car.JPEG",
               0.4, 0.4)
