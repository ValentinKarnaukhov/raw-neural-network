import math

import matplotlib.pyplot as plt
import numpy as np


def plot_error(errors):
    plt.plot(errors)
    plt.grid(True)
    plt.title("Mean square error")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.show()


def plot_images(input_dataset):
    dataset_length = len(input_dataset)
    columns = max(math.ceil(dataset_length / 5), 5)
    rows = math.ceil(dataset_length / columns)
    fig, axs = plt.subplots(rows, columns)
    data_counter = 0
    for row in range(0, rows):
        for column in range(0, columns):
            if data_counter < dataset_length:
                axs[row, column].imshow(input_dataset[data_counter])
            axs[row, column].axis('off')
            data_counter += 1
    plt.show()


def plot_predictions(input_dataset, predications, values_dictionary=None):
    dataset_length = len(input_dataset)
    p_size = predications.shape[-1]
    columns = max(math.ceil(dataset_length / 5), 5)
    rows = math.ceil(dataset_length / columns)
    fig, axs = plt.subplots(rows, columns * 2, layout="constrained")
    data_counter = 0
    if values_dictionary is None:
        ticks = np.arange(p_size)
    else:
        ticks = list(values_dictionary.values())
    for row in range(0, rows):
        for column in range(0, columns):
            if data_counter < dataset_length:
                axs[row, column * 2].imshow(input_dataset[data_counter])
                axs[row, column * 2].axis('off')
                axs[row, column * 2 + 1].barh(np.arange(p_size), predications[data_counter])
                axs[row, column * 2 + 1].yaxis.tick_right()
                axs[row, column * 2 + 1].set_yticks(np.arange(p_size), ticks)
            data_counter += 1
    plt.show()
