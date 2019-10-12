"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""


from facenet_pytorch import MTCNN
from torch_mtcnn.visualization_utils import show_bboxes
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
from model import ExpressionSqueezeNet
import torch
from PIL import ImageDraw
import PIL.ImageFont as ImageFont

try:
    font = ImageFont.truetype('arial.ttf', 20)
except IOError:
    print('error loading arial.ttf')
    font = ImageFont.load_default()

classes = ['MARAH', 'JIJIK', 'TAKUT', 'BAHAGIA', 'BIASA SAJA', 'SEDIH', 'TERKEJUT']

model = ExpressionSqueezeNet()
mtcnn = MTCNN(keep_all=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load('imfdb-squeezenet2.pth')
model.load_state_dict(state)
model.to(device)
model.eval()

valid_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def predict(model, img):
    x = valid_transforms(img).to(device)
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        _, p = torch.max(logits, 1)
    return classes[p]


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        pil = Image.fromarray(img)
        bb, prob = mtcnn.detect(pil)
        try:
          bboxed = show_bboxes(pil, bb)
          for b in bb:
              draw = ImageDraw.Draw(bboxed)
              pred = predict(model, pil.crop(b))
              draw.text((b[0], b[1]), pred, font=font)
        except Exception:
          bboxed = pil
        cv2.imshow('my webcam', np.array(bboxed))
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
