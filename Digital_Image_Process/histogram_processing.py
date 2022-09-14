import matplotlib.pyplot as plt


def plot_histogram():
    x = list(range(8))
    y = [0, 0.17, 0, 0.25, 0.21, 0, 0.23, 0.14]
    print(sum(y))
    plt.bar(x, y)
    plt.grid(axis='y', ls='-.')
    for a, b, i in zip(x, y, range(len(x))):
        if y[i] != 0:
            plt.text(a, b + 0.001, "%.2f"%y[i], ha='center')
    plt.title("historgram after historgram processing")
    plt.xlabel("gray_level")
    plt.ylabel("probability")
    plt.show()


if __name__ == "__main__":
    plot_histogram()