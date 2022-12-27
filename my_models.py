from abc import ABC, abstractmethod

import torch
from torchvision.models import ResNeXt50_32X4D_Weights


class Model(ABC):
    @abstractmethod
    def __init__(self, device):
        pass

    @abstractmethod
    def infer(self, batch):
        pass


class ResnextModel(Model):
    def __init__(self, device):
        self.device = device
        self.weights = ResNeXt50_32X4D_Weights.DEFAULT
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d',
                                    weights=self.weights).to(device)
        self.model.eval()
        self.preprocess = self.weights.transforms()

    def infer(self, batch):
        with torch.no_grad():
            output = self.model(batch.to(self.device))
        return torch.argmax(output, dim=1)


class YoloModel(Model):
    def __init__(self, device):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True).to(device)

    def infer(self, imgs):
        # TODO imgs? torch?
        # TODO less verbose
        results = self.model(imgs)

        results.save()
        # print(results.xyxy[0])  # img1 predictions (tensor)
        # print(results.pandas().xyxy[0])

        return results
