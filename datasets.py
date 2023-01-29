from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset


class ImageNetWithIndices(ImageNet):
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index

    def get_n_label(self, n):
        return self.classes[n][0]

    def get_label(self, index):
        return self.get_n_label(self.imgs[index][1])

    def get_filepath(self, index):
        return self.imgs[index][0]


class ImageNetSomeFiles(ImageFolder):
    def __init__(self, root, transform, imagenet_data):
        self.imagenet_data = imagenet_data
        super().__init__(root=root, transform=transform, target_transform=self.target_transform)

    def target_transform(self, class_idx):
        class_name = self.classes[class_idx]
        return self.imagenet_data.class_to_idx[class_name]


class TensorsDataset(Dataset):
    # loads arbitrary images to memory
    def __init__(self, tensors_w_labels, transform, imagenet_data):
        def rgb(pil):
            # required because there are some grayscale images, and they fail our transform function
            if pil.mode == 'RGB':
                return pil
            else:
                return pil.convert('RGB')

        to_pil = transforms.ToPILImage()
        self.images = [transform(rgb(to_pil(img))) for img, _ in tensors_w_labels]
        self.labels = [imagenet_data.class_to_idx[class_name] for _, class_name in tensors_w_labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# TODO del all of this
if __name__ == '__main__':
    root = 'temp'
    batch_size = 8

    from image_utils import load_persisted
    from misc import get_device
    from pathlib import Path
    import torch, torchvision

    device = get_device()

    imagenet_data, resnext, yolo = load_persisted(device, '/cs_storage/public_datasets/ImageNet')

    imagenet_dataset = ImageNetSomeFiles(root=root,
                                         transform=resnext.preprocess,
                                         imagenet_data=imagenet_data)
    x1 = imagenet_dataset[0]
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(imagenet_dataset,
                                                                           batch_size=batch_size,
                                                                           shuffle=False,
                                                                           num_workers=0)
    for X, y in data_loader:
        # print(X)
        im_x = X.to(device)
        im_y = y.to(device)
        break

    ######

    tensors_w_labels = [(torchvision.io.read_image(str(img)), img.parent.name) for img in Path(root).glob('*/*.JPEG')]
    tensor_dataset = TensorsDataset(tensors_w_labels=tensors_w_labels,
                                    transform=resnext.preprocess,
                                    imagenet_data=imagenet_data)
    x1 = tensor_dataset[0]
    data_loader = torch.utils.data.DataLoader(tensor_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)
    for X, y in data_loader:
        # print(X)
        ts_x = X.to(device)
        ts_y = y.to(device)
        break

    print(im_x, ts_x)
    print(im_y, ts_y)
    print((im_x - ts_x).count_nonzero())
