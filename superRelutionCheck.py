import cv2
import os
import matplotib.pyplot as plt
import numpy as np
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if (image is not None):
            images.append(image)

    return images


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


originPath = ""

targetFolder = ""

targetImages = load_images_from_folder(targetFolder)


originImage = cv2.imread(originPath)

MAX = 2**8-1

for i in range(0,len(targetImages)):
    plt.imshow(to_gray(targetImages[i]))
    plt.show()
    mse = compare_mse(originImage,targetImages[i])
    print("mse:{}",format(mse))
    psnr = compare_psnr(originImage,targetImages[i])
    print("msnr:{}",format(psnr))
    ssim = compare_ssim(originImage,targetImages[i])
    print("psnr:{}",format(ssim))

