import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from skimage.feature import hog
from skimage import data, exposure, io
from skimage.transform import rescale, resize
import cv2
import os
import glob

meses = ['janeiro', 'fevereiro', 'marco', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']

def hogFeatures(imag):
    image = io.imread(imag)
    resizesImage = resize(image,(60,150))
    fd = hog(resizesImage, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1), visualize=False, multichannel=False)
    return fd


def caracteristicas(treino=''):
    print(meses)
    hogs = []
    labels = []
    # r=root, d=directories, f = files
    for key, m in enumerate(meses):
        listOfFiles = [f for f in glob.glob("./"+m+"/"+treino+"*.bmp", recursive=False)]
        for img in listOfFiles:
            print(img)
            fd = hogFeatures(img)
            hogs.append(fd)
            labels.append(m)
    return(hogs,labels)

def main():
    print("Extraido caracteristicas")
    treinoFeats = []
    treinoLabels = []
    testeFeats = []
    testeLabels = []

    testeFeats, testeLabels = caracteristicas('teste/')
    treinoFeats, treinoLabels = caracteristicas()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(treinoFeats, treinoLabels)
    pred = model.score(testeFeats,testeLabels)
    print("%.02f %%"  %(pred*100))
    '''
    maio1 = io.imread("./maio/md1.bmp")
    maio2 = io.imread("./maio/md1.bmp")
    maio3 = io.imread("./maio/md1.bmp")

    janeiro1 = io.imread("./janeiro/j1.bmp")
    janeiro2 = io.imread("./janeiro/j2.bmp")
    janeiro3 = io.imread("./janeiro/j3.bmp")

    f1 = hogFeatures(maio1)
    feats.append(f1)
    labels.append("maio")

    f1 = hogFeatures(maio2)
    feats.append(f1)
    labels.append("maio")

    f1 = hogFeatures(janeiro1)
    feats.append(f1)
    labels.append("janeiro")

    f1 = hogFeatures(janeiro2)
    feats.append(f1)
    labels.append("janeiro")


    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(feats, labels)


    feat1 = hogFeatures(maio3)
    feat2 = hogFeatures(janeiro3)
    print(feat1.reshape(1, -1))
    pred = model.score((feat2.reshape(1, -1)),['maio'])
    print(pred)
    
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    '''

if __name__ == "__main__":
    main()
