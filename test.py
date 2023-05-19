import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from fl_model import Network

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "/home/ren/repos/simple_test_2/resnet_weights/deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
model_path = "/home/ren/repos/simple_test_2/resnet_weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# weights_path = '/home/ren/repos/68_39_test/face_landmarks_first_try.pth'
weights_path = '/home/ren/repos/68_39_test/face_landmarks_third_try.pth'
path = '/home/ren/repos/68_39_test/Лица'

best_network = Network()
best_network.load_state_dict(torch.load(
    weights_path, map_location=torch.device('cpu')))
best_network.eval()

# загрузим модель Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

images_path = []
for root, dirs, files in os.walk(path):
    for filename in files:
        images_path.append(path + '/' + filename)

for img in images_path:

# читаем изображение
    original_image = cv2.imread(img)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    display_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image = cv2.imread(img)
    # получаем ширину и высоту изображения
    h, w = image.shape[:2]

    # предварительная обработка: изменение размера и вычитание среднего
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # устанавливаем на вход нейронной сети изображение
    model.setInput(blob)
    # выполняем логический вывод и получаем результат
    output = np.squeeze(model.forward())

    font_scale = 1.0
    all_landmarks = []
    for i in range(0, output.shape[0]):
        # получить уверенность
        confidence = output[i, 2]
        # если достоверность выше 50%, то нарисуйте окружающий прямоугольник
        if confidence > 0.5:
            # получить координаты окружающего блока и масштабировать их до исходного изображения
            box = output[i, 3:7] * np.array([w, h, w, h])
            # преобразовать в целые числа
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # рисуем прямоугольник вокруг лица
            cv2.rectangle(display_image, (start_x, start_y), (end_x, end_y),
                          color=(255, 0, 0), thickness=15)

            temp_image = grayscale_image[start_y:end_y, start_x:end_x]
            temp_image = TF.resize(Image.fromarray(temp_image), size=(224, 224))
            temp_image = TF.to_tensor(temp_image)
            temp_image = TF.normalize(temp_image, [0.5], [0.5])
            with torch.no_grad():
                landmarks = best_network(temp_image.unsqueeze(0))
            # print(landmarks)
            landmarks = (landmarks.view(68, 2).detach().numpy() + 0.5) * \
                np.array([[end_x-start_x, end_y-start_y]]) + \
                np.array([[start_x, start_y]])
            
            # print(landmarks)
            # for i in range(0, 146, 2):
            #     if landmarks[i] < start_x or landmarks[i + 1] < start_y:
            #         landmarks = np.delete(landmarks, i)
                    
            # print(landmarks)
            # landmarks = (landmarks.view(68,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
            all_landmarks.append(landmarks)

    # show the image
    plt.imshow(display_image)
    for landmarks in all_landmarks:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='c', s=6)
    plt.show()
