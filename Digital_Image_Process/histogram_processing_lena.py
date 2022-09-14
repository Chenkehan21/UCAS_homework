from filecmp import cmp
import cv2
import matplotlib.pyplot as plt
import numpy as np


def histogram_lena():
    img = cv2.imread('./lena.jpg', 0)

    histogramed1_img = cv2.equalizeHist(img)
    histogramed2_img = cv2.equalizeHist(histogramed1_img)
    histogramed3_img = cv2.equalizeHist(histogramed2_img)
    fig = plt.figure()
    plt.suptitle("Gray image with different times of histogram processing", fontsize=20)

    ax = fig.add_subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("raw picture")
    ax =fig.add_subplot(2, 4, 5)
    plt.hist(img.flatten(), bins=256, range=[0, 256], density=True)
    plt.title("raw histogram")

    ax = fig.add_subplot(2, 4, 2)
    plt.imshow(histogramed1_img, cmap='gray')
    plt.title("onece histogram picture")
    ax = fig.add_subplot(2, 4, 6)
    plt.hist(histogramed1_img.flatten(), bins=256, range=[0, 256], density=True)
    plt.title("onece histogram processing")

    ax = fig.add_subplot(2, 4, 3)
    plt.imshow(histogramed2_img, cmap='gray')
    plt.title("twice histogram picture")
    ax = fig.add_subplot(2, 4, 7)
    plt.hist(histogramed2_img.flatten(), bins=256, range=[0, 256], density=True)
    plt.title("twice histogram processing")

    ax = fig.add_subplot(2, 4, 4)
    plt.imshow(histogramed3_img, cmap='gray')
    plt.title("three times histogram picture")
    ax = fig.add_subplot(2, 4, 8)
    plt.hist(histogramed3_img.flatten(), bins=256, range=[0, 256], density=True)
    plt.title("three histogram processing")

    plt.show()


if __name__ == "__main__":
    histogram_lena()