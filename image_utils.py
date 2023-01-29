import shutil
from pathlib import Path
import logging
import shelve
from typing import Callable

import torch
import torchvision
from torch.utils.data import Subset
from torch import Tensor, sigmoid
from PIL import Image

import models_wrapper
from datasets import ImageNetWithIndices, ImageNetSomeFiles, TensorsDataset
from misc import get_device, initial_dump_images

# required for yolo unpickling
import sys
import os.path
from torch import hub

sys.path.append(os.path.join(hub.get_dir(), 'ultralytics_yolov5_master'))
import models  # it's in the aforementioned path

logger = logging.getLogger(__name__)

SHELVE_FILE = "persist"


def load_persisted(device, imagenet_path):
    if device == 'cpu':
        global SHELVE_FILE
        SHELVE_FILE += '_cpu'
    with shelve.open(SHELVE_FILE) as shelve_db:
        if 'resnext' not in shelve_db:
            logger.debug('Creating ResNext model')
            shelve_db['resnext'] = models_wrapper.ResnextModel(device)
        if 'yolo' not in shelve_db:
            logger.debug('Creating Yolo model')
            shelve_db['yolo'] = models_wrapper.YoloModel(device)

        resnext = shelve_db['resnext']
        yolo = shelve_db['yolo']

        if 'imagenet_data' not in shelve_db:
            logger.debug('Loading ImageNet')
            shelve_db['imagenet_data'] = ImageNetWithIndices(imagenet_path,
                                                             transform=resnext.preprocess)
        imagenet_data = shelve_db['imagenet_data']
    return imagenet_data, resnext, yolo


def extract_resnext_correct(batch_size, device, imagenet_data, num_of_images, classes, num_of_images_threads, resnext):
    # keeps only images that resnext is initially correct about them, and return the indices and resnext's confidence

    if classes:
        target_indices = [i for i, target in enumerate(imagenet_data.targets) if
                          imagenet_data.get_n_label(target) in classes]
        imagenet_subset = Subset(imagenet_data, target_indices)
    else:
        logger.debug('Using all classes')
        imagenet_subset = imagenet_data

    data_loader = torch.utils.data.DataLoader(imagenet_subset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_of_images_threads)
    images_indices = []
    probs = torch.Tensor().to(device)
    for X, y, indices in data_loader:
        X = X.to(device)
        y = y.to(device)
        prob, y_hat = resnext.infer(X)
        if torch.equal(y, y_hat):
            images_indices += indices
            probs = torch.cat((probs, prob))
        else:
            for i in range(len(indices)):
                if y[i] == y_hat[i]:
                    images_indices.append(indices[i])
                    probs = torch.cat((probs, prob[i].unsqueeze(dim=0)))
        if len(images_indices) >= num_of_images * 4:  # we'll throw many images because of YOLO, so take more initially
            break
    assert len(images_indices) == len(probs)
    return images_indices, probs


def extract_yolo(imgs_with_labels, num_of_images, probs, threshold_confidence, threshold_size_ratio, yolo):
    # keep only images that yolo is confident about them, and get its results
    yolo_results = yolo.infer([img for img, label in imgs_with_labels])
    image_results = []
    image_probs = []
    for i, (img, img_label) in enumerate(imgs_with_labels):
        assert Path(img).stem == Path(yolo_results[i]['file']).stem
        img_shape = yolo_results[i]['im'].shape
        img_area = img_shape[0] * img_shape[1]
        keep = False
        for x1, y1, x2, y2, confidence, label in yolo_results[i]['xyxy']:
            width_x = x2 - x1
            width_y = y2 - y1
            if width_x * width_y / img_area > threshold_size_ratio and confidence > threshold_confidence:
                keep = True
                break
        if keep:
            image_results.append({
                'img': img,
                'label': img_label,
                'bb': yolo_results[i]['xyxy']
            })
            image_probs.append(probs[i])

        if len(image_results) >= num_of_images:
            break
    return image_probs, image_results


def prepare(num_of_images_threads, imagenet_path, batch_size, num_of_images, classes, random_seed,
            threshold_size_ratio, threshold_confidence):
    torch.manual_seed(random_seed)
    device = get_device()

    shutil.rmtree('runs', ignore_errors=True)

    imagenet_data, resnext, yolo = load_persisted(device, imagenet_path)

    images_indices, probs = extract_resnext_correct(batch_size, device, imagenet_data, num_of_images, classes,
                                                    num_of_images_threads, resnext)

    imgs_with_labels = [(imagenet_data.get_filepath(i), imagenet_data.get_label(i)) for i in images_indices]

    image_probs, image_results = extract_yolo(imgs_with_labels, num_of_images, probs, threshold_confidence,
                                              threshold_size_ratio, yolo)

    tensors_w_labels = [(torchvision.io.read_image(im['img']), im['label']) for im in image_results]
    tensors_dataset = TensorsDataset(tensors_w_labels=tensors_w_labels,
                                     transform=resnext.preprocess,
                                     imagenet_data=imagenet_data)

    initial_dump_images(image_results)

    logger.debug(f"Running on {len(image_results)} images: {[im['img'] for im in image_results]}")

    return {
        'device': device,
        'resnext': resnext,
        'imagenet_data': imagenet_data,
        'image_results': image_results,
        'image_probs': torch.Tensor(image_probs).to(device),
        'tensors_w_labels': tensors_w_labels,
    }


def infer_images(tensors_w_labels, model, batch_size, num_of_images_threads, imagenet_data):
    dataset = TensorsDataset(tensors_w_labels=tensors_w_labels,
                             transform=model.preprocess,
                             imagenet_data=imagenet_data)
    try:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  # TODO increasing num_workers to num_of_images_threads seems to cause problems
                                                  num_workers=0)
        all_prob = torch.Tensor().to(model.device)
        all_y = torch.Tensor().to(model.device)
        all_y_hat = torch.Tensor().to(model.device)
        for X, y in data_loader:
            X = X.to(model.device)
            y = y.to(model.device)
            prob, y_hat = model.infer(X)
            all_prob = torch.cat((all_prob, prob))
            all_y = torch.cat((all_y, y))
            all_y_hat = torch.cat((all_y_hat, y_hat))
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"torch.cuda.OutOfMemoryError - reducing batch size to {batch_size // 2}")
        all_y, all_y_hat, all_prob = infer_images(dataset, model, batch_size=batch_size // 2,
                                                  num_of_images_threads=num_of_images_threads)

    return all_y, all_y_hat, all_prob


def get_patch(func, width_x, width_y, dominant_color, device):
    yy, xx = torch.meshgrid(torch.arange(width_y), torch.arange(width_x), indexing='ij')
    xx = xx.to(device)
    yy = yy.to(device)
    result = func(x=xx, y=yy)
    if not isinstance(result, torch.Tensor) or not result.shape:
        assert type(result) in [float, torch.Tensor]
        result = torch.full_like(xx, result, dtype=torch.float)
    result = sigmoid(result)
    result = (result > 0.5).to(torch.uint8)
    result = result.unsqueeze(dim=0)
    result = torch.concatenate(
        (result * dominant_color[0], result * dominant_color[1], result * dominant_color[2]), axis=0)
    result[result == 0] = 255
    return result


def get_dominant_color(im):
    transform = torchvision.transforms.ToPILImage()
    pil_image = transform(im)
    pil_image = pil_image.convert("RGBA")
    pil_image = pil_image.resize((1, 1), resample=0)
    return pil_image.getpixel((0, 0))


def apply_patches(func: Callable[[Tensor, Tensor], Tensor], im: Tensor, xyxy: Tensor, ratio_x: float, ratio_y: float,
                  device: str) -> None:
    for x1, y1, x2, y2, confidence, label in xyxy:
        width_x = int(x2 - x1)
        width_y = int(y2 - y1)
        patch_width_x = int(width_x * ratio_x)
        patch_width_y = int(width_y * ratio_y)
        start_x = int(x1 + (width_x - patch_width_x) / 2)
        start_y = int(y1 + (width_y - patch_width_y) / 2)
        dominant_color = get_dominant_color(im[:, start_y:start_y + patch_width_y, start_x:start_x + patch_width_x])
        patch = get_patch(func, patch_width_x, patch_width_y, dominant_color, device)

        if im.shape[0] == 3:
            im[:, start_y:start_y + patch_width_y, start_x:start_x + patch_width_x] = patch
        elif im.shape[0] == 1:
            # logger.debug(f'Image {img} is grayscale')     # multiple printings...
            im[:, start_y:start_y + patch_width_y, start_x:start_x + patch_width_x] = patch[0]
        else:
            raise ValueError
