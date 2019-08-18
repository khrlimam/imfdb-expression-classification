from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
topil = transforms.ToPILImage()
totensor = transforms.Compose(valid_transforms.transforms[:-1])


def get_train_valid_data(batchsize):
    train_set = ImageFolder(root='/home/khairulimam/datasets/expressions/IMFDB/train/', transform=train_transforms)
    valid_set = ImageFolder(root='/home/khairulimam/datasets/expressions/IMFDB/valid/', transform=valid_transforms)
    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batchsize, shuffle=True)

    return train_loader, valid_loader
