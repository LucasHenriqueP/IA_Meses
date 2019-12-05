import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from skimage.feature import hog
from skimage import data, exposure, io
from skimage.transform import rescale, resize
import cv2
import os


meses = ['janeiro', 'fevereiro', 'marco', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']

def hogFeatures(image):
    #image = io.imread(imag)
    resizesImage = resize(image,(60,150))
    fd = hog(resizesImage, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1), visualize=False, multichannel=False)
    print(fd)
    return fd


def caracteristicas():
    print(meses)
    hogs = []
    labels = []
    # r=root, d=directories, f = files
    saida = open('output.txt','w')
    for key, m in enumerate(meses):
        for r, d, f in os.walk(m):
            for file in f:
                if '.bmp' in file:
                    fd = hogFeatures(os.path.join(r, file))
        hogs.append(fd)
        labels.append(m)
    print(len(hogs))

def main():
    print("Extraido caracteristicas")
    #caracteristicas()
    feats = []
    labels = []
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


    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(feats, labels)


    feat1 = hogFeatures(maio3)
    feat2 = hogFeatures(janeiro3)
    pred = model.predict(feat1.reshape(1, -1))[0]
    print(pred)
    maio3 = resize(maio3,(60,150))
    maio3 = cv2.cvtColor(maio3,cv2.COLOR_GRAY2RGB)
    cv2.putText(maio3, pred.title(), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		(255, 0, 0), 3)
    cv2.imshow("Test Image #{}".format(0 + 1), maio3)
    cv2.waitKey(0)
    
    '''
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
