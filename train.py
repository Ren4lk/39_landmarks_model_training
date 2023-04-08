import time
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import dataset
import transforms
import fl_model


def printStepInFile(file_name, epoch, step, total_step, loss, operation):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name), 'a') as file:
        if operation == 'train':
            file.write("Epoch: %d  Train Steps: %d/%d  Loss: %.4f \n" %
                       (epoch, step, total_step, loss))
        elif operation == 'valid':
            file.write("Epoch: %d  Valid Steps: %d/%d  Loss: %.4f \n" %
                       (epoch, step, total_step, loss))


def printEpochInfoInFile(file, epoch, loss_valid, loss_train=None, total_epochs=None):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), file), 'a') as file:
        if loss_train:
            file.write('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}\n'.format(
                epoch, loss_train, loss_valid))
        else:
            file.write(
                "Minimum Validation Loss of {:.4f} at epoch {}/{}\nModel Saved!\n".format(loss_valid, epoch, total_epochs))


if __name__ == '__main__':
    ds = dataset.ProfileLandmarksDataset(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'annotation_file.txt'), transform=transforms.Transforms())
    len_valid_set = int(0.1*len(ds))
    len_train_set = len(ds) - len_valid_set

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset,  = torch.utils.data.random_split(
        ds, [len_train_set, len_valid_set])

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = DataLoader(
        valid_dataset, batch_size=64, shuffle=True, num_workers=4)

    torch.autograd.set_detect_anomaly(True)
    network = fl_model.Network().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    loss_min = np.inf
    NUM_EPOCHS = 1000

    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS+1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step in range(1, len(train_loader)+1):
            images, landmarks = next(iter(train_loader))
            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0), -1).cuda()
            predictions = network(images)
            optimizer.zero_grad()
            loss_train_step = criterion(predictions, landmarks)
            loss_train_step.backward()
            optimizer.step()
            loss_train += loss_train_step.item()
            running_loss = loss_train/step

            printStepInFile('train_info.txt',
                            epoch,
                            step,
                            len(train_loader),
                            running_loss,
                            'train')

        network.eval()
        with torch.no_grad():
            for step in range(1, len(valid_loader)+1):
                images, landmarks = next(iter(valid_loader))
                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0), -1).cuda()
                predictions = network(images)
                loss_valid_step = criterion(predictions, landmarks)
                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step

                printStepInFile('train_info.txt',
                                epoch,
                                step,
                                len(valid_loader),
                                running_loss,
                                'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        printEpochInfoInFile('train_info.txt',
                             epoch,
                             loss_valid,
                             loss_train)

        if loss_valid < loss_min:
            loss_min = loss_valid
            printEpochInfoInFile('train_info.txt',
                                 epoch,
                                 loss_valid,
                                 total_epochs=NUM_EPOCHS)
            torch.save(network.state_dict(), os.path.join(
                os.path.dirname(os.path.abspath(__file__)), f'face_39_landmarks_weights.pth'))

        print('Training Complete')
        print("Total Elapsed Time : {} s".format(time.time()-start_time))
