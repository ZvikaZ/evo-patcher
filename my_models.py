# TODO change this file's name

from abc import ABC, abstractmethod

import numpy as np
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
    # Yolo guide: https://github.com/ultralytics/yolov5/issues/36
    def __init__(self, device):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, _verbose=False).to(device)
        self.batch_size = 100

    def infer(self, imgs):
        yolo_results = []
        for chunk in np.array_split(imgs, len(imgs) / self.batch_size + 1):
            chunk_results = self.model(chunk.tolist())
            chunk_results.save()
            yolo_results.extend(chunk_results.tolist())
        results = []
        for result in yolo_results:
            assert result.n == len(result.files) == len(result.ims) == len(result.xyxy) == 1
            results.append({
                'file': result.files[0],
                'im': result.ims[0],
                'xyxy': result.xyxy[0],
            })
        return results
