# TODO clean all commented prints
from pathlib import Path

import torch

import models_wrapper
from datasets import ImageNetWithIndices, ImageNetSomeFiles
from misc import get_device, dump_images


def infer_images(root, model, imagenet_data, batch_size, num_of_images_threads):
    dataset = ImageNetSomeFiles(root=root,
                                transform=model.preprocess,
                                imagenet_data=imagenet_data)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              # TODO increasing num_workers to num_of_images_threads seems to cause problems
                                              num_workers=0)
    success = 0
    fail = 0
    for X, y in data_loader:
        X = X.to(model.device)
        y = y.to(model.device)
        y_hat = model.infer(X)

        # TODO vectorize
        if torch.equal(y, y_hat):
            # print('OK:', y, y_hat)
            success += len(y)
        else:
            for i in range(len(y)):
                if y[i] == y_hat[i]:
                    # print('OK i:', i, y[i], y_hat[i])
                    success += 1
                else:
                    fail += 1
                    # print(f"mismatch. expected: {y[i]}, " +
                    #       f"predicted: {y_hat[i]}")
    assert success + fail == len(dataset)
    return success / len(dataset)
    # TODO return (some?) mismatches for demonstration?


def prepare(num_of_images_threads, imagenet_path, batch_size, num_of_images, threshold_size_ratio, threshold_confidence):
    torch.manual_seed(1)  # TODO move to main.py (and suggest removing)
    device = get_device()

    resnext = models_wrapper.ResnextModel(device)
    yolo = models_wrapper.YoloModel(device)

    imagenet_data = ImageNetWithIndices(imagenet_path,
                                        transform=resnext.preprocess)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_of_images_threads)

    images_indices = []
    for X, y, indices in data_loader:
        X = X.to(device)
        y = y.to(device)
        y_hat = resnext.infer(X)
        if torch.equal(y, y_hat):
            images_indices += indices
        else:
            for i in range(len(indices)):
                if y[i] == y_hat[i]:
                    images_indices.append(indices[i])
        if len(images_indices) >= num_of_images * 4:  # we'll throw many images because of YOLO, so take more initially
            break

    imgs_with_labels = [(imagenet_data.get_filepath(i), imagenet_data.get_label(i)) for i in images_indices]
    dump_images(imgs_with_labels, "initial")

    yolo_results = yolo.infer([img for img, label in imgs_with_labels])
    image_results = []
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

        if len(image_results) >= num_of_images:
            break

    return {
        'device': device,
        'resnext': resnext,
        'imagenet_data': imagenet_data,
        'image_results': image_results,
    }
