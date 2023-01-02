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
        self.softmax = torch.nn.Softmax(dim=1)

    def infer(self, batch):
        with torch.no_grad():
            logits = self.model(batch.to(self.device))
        output = self.softmax(logits)
        prob, y_hat = output.topk(k=1)
        assert prob.shape[-1] == y_hat.shape[-1] == 1
        return prob.squeeze(), y_hat.squeeze()


class YoloModel(Model):
    # Yolo guide: https://github.com/ultralytics/yolov5/issues/36
    def __init__(self, device):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, _verbose=False).to(device)
        self.batch_size = 100

    def infer(self, imgs):
        yolo_results = []
        for chunk in np.array_split(imgs, len(imgs) / self.batch_size + 1):
            chunk_results = self.model(chunk.tolist())
            chunk_results.save()  # TODO disable the saving printing, and maybe unify the directories
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
