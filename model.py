import torch
from torchvision.models import MobileNetV2
from torchvision.models.mobilenet import model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class ExpressionClassifier(MobileNetV2):
    def __init__(self, pretrained=True, progress=True, num_classes=1000):
        super().__init__()
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=progress)
            self.load_state_dict(state_dict)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(.2),
            torch.nn.Linear(62720, num_classes),
            torch.nn.LogSoftmax(dim=1)  # use NLLLoss as the criterion
        )

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.classifier(x)
