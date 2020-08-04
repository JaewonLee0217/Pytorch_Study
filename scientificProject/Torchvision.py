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
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

#Load Image
img = Image.open('img/img07.png') #Image로드 할 떄는 Pillow패키지의 Image.open()을 이용해서 im
img = img.resize((IMG_SIZE, int(img.height*IMG_SIZE/img.width)))
#가로 세로 비율에 맞추어 resize해준다.
plt.figure(figsize=(16,16))
plt.imshow(img)

#Image to Tensor
trf = T.Compose([
    T.ToTensor()
])
#T.compose를 하는 게 이 리스트 안에 있는 작업들을 차례로 수행한다.
#T.ToTensor : 이미지를 0-1사이의 값을 가지는 텐서로 바꾼다.
input_img = trf(img)

print(input_img.shape)

out = model([input_img])[0]
print(out.keys())
#boxes:사람의 박스 영역(X0,Y0,X1,Y1)
#socres:bounding box를 사람이라고 판단한 점수
#keypoints: 사람의 키포인트 점들의 좌표

####################여기 까지가 모델을 통해 분석을 마친 것이고 out이 있긴 하지만 invisual상태니까
####################이제부터는 visualize해준다.
codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]

fig, ax = plt.subplots(1, figsize=(16, 16))
ax.imshow(img)

for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
    score = score.detach().numpy()

    if score < THRESHOLD:
        continue

    box = box.detach().numpy()
    keypoints = keypoints.detach().numpy()[:, :2]

    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='b',
                             facecolor='none')
    ax.add_patch(rect)

    # 17 keypoints
    for k in keypoints:
        circle = patches.Circle((k[0], k[1]), radius=2, facecolor='r')
        ax.add_patch(circle)

    # draw path
    # left arm
    path = Path(keypoints[5:10:2], codes)
    line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    ax.add_patch(line)

    # right arm
    path = Path(keypoints[6:11:2], codes)
    line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    ax.add_patch(line)

    # left leg
    path = Path(keypoints[11:16:2], codes)
    line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    ax.add_patch(line)

    # right leg
    path = Path(keypoints[12:17:2], codes)
    line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    ax.add_patch(line)





