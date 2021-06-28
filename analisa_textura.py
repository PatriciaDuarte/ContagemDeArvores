import os
import csv
import math
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt

dataset_path = "textura_recorte"
classes = ["centro", "naocentro"]

image_types = ['jpg', 'JPG', 'png']


for classe in classes:
	xs = []
	ys = []
	file_names = [fn for fn in os.listdir(os.path.join(dataset_path,classe)) if any(fn.endswith(ext) for ext in image_types)]
	for file in file_names:
                print("Processando arquivo ",file, " da classe", classe)
                filename, file_extension = os.path.splitext(file) 
                temp = os.path.join(dataset_path, classe, file)
                print(temp)
                J = cv2.imread(temp)
                grayImage = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
                glcm = greycomatrix(grayImage, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
                xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
                ys.append(greycoprops(glcm, 'correlation')[0, 0])

                if classe == 'centro':
                     plt.plot(xs, ys, 'go', label='centro')
                else:
                     plt.plot(xs, ys, 'bo', label='naocentro') 
plt.xlabel('GLCM Dissimilarity')
plt.ylabel('GLCM Correlation') 
plt.show()
print("testando")