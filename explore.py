# TODO change this file's name
# TODO clean all commented prints
from pathlib import Path

import torch

import my_models
from datasets import ImageNetWithIndices, ImageNetSomeFiles
from misc import get_device, dump_images

IMAGENET_PATH = '/cs_storage/public_datasets/ImageNet'
NUM_OF_THREADS = 8  # 0 is disabled
BATCH_SIZE = 40  # TODO
NUM_OF_IMAGES = 3  # TODO 30

YOLO_THRESHOLD_SIZE_RATIO = 0.1
YOLO_THRESHOLD_CONFIDENCE = 0.8


def infer_images(root, model, imagenet_data):
    dataset = ImageNetSomeFiles(root=root,
                                transform=model.preprocess,
                                imagenet_data=imagenet_data)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              # TODO increasing num_workers to NUM_OF_THREADS seems to cause problems
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


def prepare():
    torch.manual_seed(1)  # TODO remove
    device = get_device()

    resnext = my_models.ResnextModel(device)
    yolo = my_models.YoloModel(device)

    imagenet_data = ImageNetWithIndices(IMAGENET_PATH,
                                        transform=resnext.preprocess)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=NUM_OF_THREADS)

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
        if len(images_indices) >= NUM_OF_IMAGES * 4:  # we'll throw many images because of YOLO, so take more initially
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
            if width_x * width_y / img_area > YOLO_THRESHOLD_SIZE_RATIO and confidence > YOLO_THRESHOLD_CONFIDENCE:
                keep = True
                break
        if keep:
            image_results.append({
                'img': img,
                'label': img_label,
                'bb': yolo_results[i]['xyxy']
            })

        if len(image_results) >= NUM_OF_IMAGES:
            break

    return {
        'device': device,
        'resnext': resnext,
        'imagenet_data': imagenet_data,
        'image_results': image_results,
    }


if __name__ == '__main__':
    data = prepare()
    for i in range(1665, 1685):
        root = f'runs.1/dump/gen_69_ind_{i}'
        print(i, infer_images(root, data['resnext'], data['imagenet_data']))
