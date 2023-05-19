import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random

import transforms


class ProfileLandmarksDataset(Dataset):

    def __init__(self, annotation_file, transform=None):
        self.images = []
        self.landmarks = []
        self.crops = []
        self.transform = transform

        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                data = l.split()
                self.images.append(data[0])
                self.crops.append([float(x) for x in data[1:5]])
                points = [float(x) for x in data[5:]]
                self.landmarks.append(list(list(t)
                                      for t in zip(points[0::2], points[1::2])))

        assert len(self.images) == len(self.landmarks)

    def __len__(self):
        return len(self.images)

    def __flipLandmarks(self, points, image):
        _, w = image.shape[:2]
        result = [[w-x, y] for x, y in points]
        return result

    def __flipCrops(self, crops, image):
        _, w = image.shape[:2]
        x1 = crops[0]
        y1 = crops[1]
        x2 = crops[2]
        y2 = crops[3]
        return [w - x2, y1, w - x1, y2]

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], 0)
        crops = self.crops[idx]
        landmarks = self.landmarks[idx]

        if self.images[idx].find('left') != -1:
            image = cv2.flip(image, 1)
            crops = self.__flipCrops(crops, image)
            landmarks = self.__flipLandmarks(landmarks, image)

        if self.transform:
            image, landmarks = self.transform(
                image, landmarks, crops)

        landmarks = landmarks - 0.5

        return image, landmarks


if __name__ == '__main__':
    dataset = ProfileLandmarksDataset(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'annotation_file.txt'), transform=transforms.Transforms())

    while True:
        image, landmarks = dataset[random.randint(0, len(dataset))]
        landmarks = (landmarks + 0.5) * 300
        plt.figure(figsize=(10, 10))
        plt.imshow(image.numpy().squeeze(), cmap='gray')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=8)
        plt.show()
