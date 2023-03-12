import math

import matplotlib.pyplot as plt
import numpy as np


def plot_error(errors):
    plt.plot(errors)
    plt.grid(True)
    plt.title("Mean square error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()


def plot_digit(input_dataset, prediction):
    input_dataset = input_dataset.reshape(-1, 28, 28)
    dataset_length = len(input_dataset)
    columns = max(math.ceil(dataset_length / 5), 5)
    rows = math.ceil(dataset_length / columns)
    fig, axs = plt.subplots(rows * 2, columns)
    data_counter = 0
    for column_index in range(0, columns):
        for row_index in range(0, rows):
            axs[row_index * 2, column_index].imshow(input_dataset[data_counter])
            axs[row_index * 2 + 1, column_index].bar(np.arange(10), prediction[data_counter])
            axs[row_index * 2 + 1, column_index].set_xticks(np.arange(10))
            data_counter += 1
        if data_counter >= dataset_length:
            break
    plt.show()
