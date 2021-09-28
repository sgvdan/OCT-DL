from matplotlib import pyplot as plt


def imshow(image, title):
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()