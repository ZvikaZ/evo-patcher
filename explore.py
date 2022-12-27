# TODO replace this file's name

import torch

import my_models  # TODO better name!
from datasets import ImageNetWithIndices, ImageNetSomeFiles  # TODO, FilesDataset
from misc import get_device, dump_images

IMAGENET_PATH = '/cs_storage/public_datasets/ImageNet'
NUM_OF_THREADS = 8  # 0 - disabled ; bigger value causes 'print's to stuck in debugging
BATCH_SIZE = 4
NUM_OF_IMAGES = 30


def infer_images(root, model, imagenet_data):
    dataset = ImageNetSomeFiles(root=root,
                                transform=model.preprocess,
                                imagenet_data=imagenet_data)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=0)  # TODO NUM_OF_THREADS)
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
                else:  # TODO remove print
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
        # print(f"Shape of X [N, C, H, W]: {X.shape}")
        # print(f"Shape of y: {y.shape} {y.dtype}")
        y_hat = resnext.infer(X)
        if torch.equal(y, y_hat):
            images_indices += indices
        else:
            for i in range(len(indices)):
                if y[i] == y_hat[i]:
                    images_indices.append(indices[i])
                # else:  # TODO del
                #     print(f"mismatch. expected: {imagenet_data.get_n_label(y[i])}, " +
                #           f"predicted: {imagenet_data.get_n_label(y_hat[i])}. file: {imagenet_data.get_filepath(indices[i])}")
        if len(images_indices) >= NUM_OF_IMAGES:
            if len(images_indices) > NUM_OF_IMAGES:
                images_indices = images_indices[:NUM_OF_IMAGES]
            break

    imgs_with_labels = [(imagenet_data.get_filepath(i), imagenet_data.get_label(i)) for i in images_indices]
    dump_images(imgs_with_labels, "initial")

    bounding_boxes = yolo.infer([img for img, label in imgs_with_labels])  # TODO: less verbose

    return {
        'device': device,
        'resnext': resnext,
        'imagenet_data': imagenet_data,
        'imgs_with_labels': imgs_with_labels,
        'bounding_boxes': bounding_boxes,
    }


if __name__ == '__main__':
    print(prepare())
