import shutil
from pathlib import Path
import logging
import shelve

import torch
from torch.utils.data import Subset

import models_wrapper
from datasets import ImageNetWithIndices, ImageNetSomeFiles
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
    with shelve.open(SHELVE_FILE) as shelve_db:
        if 'resnext' not in shelve_db:
            logger.debug('Creating ResNext model')
            shelve_db['resnext'] = models_wrapper.ResnextModel(device)
        if 'yolo' not in shelve_db:
            logger.debug('Creating Yolo model')
            shelve_db['yolo'] = models_wrapper.YoloModel(device)
        if 'mask_rcnn' not in shelve_db:
            logger.debug('Creating Mask RCNN model')
            shelve_db['mask_rcnn'] = models_wrapper.MaskRCNN(device)

        resnext = shelve_db['resnext']
        yolo = shelve_db['yolo']
        mask_rcnn = shelve_db['mask_rcnn']  # TODO either use or remove

        if 'imagenet_data' not in shelve_db:
            logger.debug('Loading ImageNet')
            shelve_db['imagenet_data'] = ImageNetWithIndices(imagenet_path,
                                                             transform=resnext.preprocess)
        imagenet_data = shelve_db['imagenet_data']
    return imagenet_data, resnext, yolo, mask_rcnn


def extract_resnext_correct(batch_size, device, imagenet_data, num_of_images, classes, num_of_images_threads, resnext):
    # keeps only images that resnext is initially correct about them, and return the indices and resnext's confidence

    target_indices = [i for i, target in enumerate(imagenet_data.targets) if imagenet_data.get_n_label(target) in classes]
    imagenet_subset = Subset(imagenet_data, target_indices)

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

    imagenet_data, resnext, yolo, mask_rcnn = load_persisted(device, imagenet_path)

    images_indices, probs = extract_resnext_correct(batch_size, device, imagenet_data, num_of_images, classes,
                                                    num_of_images_threads, resnext)

    imgs_with_labels = [(imagenet_data.get_filepath(i), imagenet_data.get_label(i)) for i in images_indices]

    image_probs, image_results = extract_yolo(imgs_with_labels, num_of_images, probs, threshold_confidence,
                                              threshold_size_ratio, yolo)

    initial_dump_images(image_results)

    logger.debug(f"Running on {len(image_results)} images: {[im['img'] for im in image_results]}")

    return {
        'device': device,
        'resnext': resnext,
        'imagenet_data': imagenet_data,
        'image_results': image_results,
        'image_probs': torch.Tensor(image_probs).to(device),
    }


def infer_images(root, model, imagenet_data, batch_size, num_of_images_threads):
    dataset = ImageNetSomeFiles(root=root,
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
        all_y, all_y_hat, all_prob = infer_images(root, model, imagenet_data,
                                                  batch_size=batch_size // 2,
                                                  num_of_images_threads=num_of_images_threads)

    return all_y, all_y_hat, all_prob
