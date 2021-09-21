import cv2
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image, ImageDraw


def negative(image):
    draw = ImageDraw.Draw(image)
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            r, g, b = image.getpixel((x, y))
            draw.point((x, y), (255 - r, 255 - g, 255 - b))
    del draw
    return image


def high_filter(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            H = (img[i][j] * 9) - (img[i - 1][j - 1] + img[i - 1][j] + img[i - 1][j + 1] + img[i][j - 1] + img[i][j + 1] + img[i+1][j - 1] + img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = min(255, max(0, H))
    return container


def sobel_operator(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2 * img[i][j - 1] + img[i + 1][j - 1]) - (
                    img[i - 1][j + 1] + 2 * img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2 * img[i - 1][j] + img[i - 1][j + 1]) - (
                    img[i + 1][j - 1] + 2 * img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = min(255, np.sqrt(gx ** 2 + gy ** 2))
    return container


def show_hist(img, name):
    color = ("b", "g", "r")
    for i, color in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plot.title(name)
        plot.xlabel("Bins")
        plot.ylabel("num of perlex")
        plot.plot(hist, color=color)
        plot.xlim([0, 260])
    plot.show()


# Warning: high filter, sobel operator and show_hist function works with cv2.
# negative function works with standard Image library


key_image_cv = cv2.imread('me.jpeg')
img = cv2.cvtColor(key_image_cv, cv2.COLOR_BGR2GRAY)
key_img_neg = high_filter(img)
key_img_neg = cv2.cvtColor(key_img_neg, cv2.COLOR_GRAY2RGB)
plot.imshow(key_img_neg, interpolation='nearest')
plot.show()
cv2.imwrite('me_high_filter.jpeg', key_img_neg)
