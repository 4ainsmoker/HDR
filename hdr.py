import os
import argparse
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'text.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(1 / float(tokens[1]))
    return images, np.asarray(times, dtype=np.float32)


def crf(response):
    # h = semilogy([0,255], np.reshape(response, 256, []))
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = response[i][:]
        plt.semilogy(hist, color=col)
        plt.xlim([0, 256])
    plt.show()


def contrast(L):
    Lmean = np.mean(L)
    first = 1 * np.sum((L - Lmean))


# насыщенность
def saturation(img):
    satIm = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    saturation = np.mean(satIm[:, :, 1])
    return saturation

makeHDR = 1

if makeHDR == 1:
    args = input("Name of dir with SDR images: ")
    images, times = loadExposureSeq(args)
    alignMTB = cv.createAlignMTB(6, 4, True)
    alignMTB.process(images, images[1])
    calibrate = cv.createCalibrateDebevec()
    response = calibrate.process(images, times)
    # crf(response)
    merge_debevec = cv.createMergeDebevec()
    hdr = merge_debevec.process(images, times, response)
    cv.imwrite(input("Save HDR file as: "), hdr)
else:
    filename = input("Name of HDR file: ")
    hdr = cv.imread(filename, cv.IMREAD_UNCHANGED)