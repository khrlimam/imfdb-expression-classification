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
            torch.nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
