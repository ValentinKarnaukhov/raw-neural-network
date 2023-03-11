import matplotlib.pyplot as plt


def plot_error(errors):
    plt.plot(errors)
    plt.grid(True)
    plt.title("Mean square error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()

