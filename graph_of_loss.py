import matplotlib.pyplot as plt

with open('39_landmarks_model_training/train_info.txt', 'r') as f:
    train_loss = []
    valid_loss = []
    for line in f:
        if line.find('Train Loss') != -1 and line.find('Valid Loss') != -1:
            train_loss.append(float(line.split()[4]))
            valid_loss.append(float(line.split()[7]))

    plt.plot(train_loss, 'b')
    plt.plot(valid_loss, 'r')
    plt.show()
