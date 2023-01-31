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
    # TODO del
    def __init__(self, root, transform, imagenet_data):
        self.imagenet_data = imagenet_data
        super().__init__(root=root, transform=transform, target_transform=self.my_target_transform)

    def my_target_transform(self, class_idx):
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
