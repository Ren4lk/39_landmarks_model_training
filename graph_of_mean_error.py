import matplotlib.pyplot as plt

with open('39_landmarks_model_training/mean_error_info.txt', 'r') as f:
    mean_errors = []
    for line in f:
        mean_errors.append(float(line.split()[4]))

    plt.plot(mean_errors, 'b')
    plt.show()
