import numpy as np
import torch
import cv2

import fl_model

import torchvision.transforms.functional as TF
from PIL import Image


def imageToTensor(img: np.ndarray,
                  size: tuple[int, int]) -> torch.Tensor:
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = TF.resize(Image.fromarray(temp), size=size)
    temp = TF.to_tensor(temp)
    temp = TF.normalize(temp, [0.5], [0.5])
    return temp


if __name__ == '__main__':
    images = []
    crops = []
    landmarks = []

    with open('/home/renat/repos/39_landmarks_model_training/annotation_file.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            data = l.split()
            images.append(data[0])
            crops.append([float(x) for x in data[1:5]])
            points = [float(x) for x in data[5:]]
            landmarks.append(list(list(t)
                                  for t in zip(points[0::2], points[1::2])))
    network = fl_model.Network()
    network.load_state_dict(torch.load(
        '/home/renat/repos/39_landmarks_model_training/resnet18_not_pretrained_face_39_landmarks_weights_900_epochs.pth'))
    network.eval()

    total_error = 0
    num_landmarks = 0

    for img_path, box, true_landmarks in zip(images, crops, landmarks):
        if img_path.find('right') != -1:
            image = cv2.imread(img_path)
            true_landmarks = np.array(true_landmarks)

            temp_img = image[int(box[1]):int(box[3]),
                             int(box[0]):int(box[2])]
            tensor_img = imageToTensor(img=temp_img, size=(224, 224))
            tensor_img = tensor_img

            with torch.no_grad():
                predicted_landmarks = network(tensor_img.unsqueeze(0))

            h, w = temp_img.shape[:2]
            predicted_landmarks = (predicted_landmarks.view(39, 2).detach().numpy() + 0.5) * \
                np.array([[w, h]]) + np.array([[int(box[0]), int(box[1])]])

            landmark_error = np.linalg.norm(
                predicted_landmarks - true_landmarks, axis=1)

            total_error += np.sum(landmark_error)
            num_landmarks += len(landmark_error)

    mean_error = total_error / num_landmarks
    # mean_error_normalized = mean_error / mean_distance
    print(mean_error)
    # print("NME:", mean_error_normalized)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], s=8, c='red')
    # plt.scatter(true_landmarks[:, 0], true_landmarks[:, 1], s=8, c='green')
    # plt.show()
