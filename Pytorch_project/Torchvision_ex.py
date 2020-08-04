import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
#딥러닝 프레임 워크 pytorch랑 torchvision 라이브러리 사용

#부가적으로
import numpy as np
from PIL import Image
#이미지 다루는 Pillow
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
#그림으로 보여주는 matplotlib

IMG_SIZE = 480
THRESHOLD = 0.95 # 판단하는 임계값

#미리 학습된 keypoint R-CNN 모델을 다운로드 받는다.
#model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

#Load Image
img = Image.open('img/img07.png') #Image로드 할 떄는 Pillow패키지의 Image.open()을 이용해서 im
img = img.resize((IMG_SIZE, int(img.height*IMG_SIZE/img.width)))
#가로 세로 비율에 맞추어 resize해준다.
plt.figure(figsize=(16,16))
plt.imshow(img)
plt.isinteractive()
