from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder


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
