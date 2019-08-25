import torch
from torchvision.models import MobileNetV2
from torchvision.models.mobilenet import model_urls
from torchvision.models.squeezenet import SqueezeNet, model_urls as squeezenet_model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class FreezableLayers(torch.nn.Module):
    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False


class ExpressionMobileNet(MobileNetV2, FreezableLayers):
    def __init__(self, pretrained=True, progress=True, num_classes=7):
        super().__init__()
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
            self.load_state_dict(state_dict)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(.2),
            torch.nn.Linear(self.last_channel, num_classes)
        )


class ExpressionSqueezeNet(SqueezeNet, FreezableLayers):
    def __init__(self, version='1_1', num_classes=7, pretrained=True, progress=True):
        super().__init__(version=version)
        if pretrained:
            state_dict = load_state_dict_from_url(squeezenet_model_urls['squeezenet' + version], progress=progress)
            self.load_state_dict(state_dict)

        self.num_classes = num_classes
        final_conv = torch.nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            final_conv,
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
