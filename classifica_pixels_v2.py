import os
import csv
import math
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from skimage.feature import greycomatrix, greycoprops, hog, local_binary_pattern
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset_path = "textura_recorte"
classes = ["centro", "naocentro"]
path_saida = "dataset_features"

image_types = ['jpg', 'JPG', 'png']

# LE CARACTERISTICAS EXTRAIDAS
featureglcm_dataset = np.load(os.path.join(path_saida,"featureGLCM_BP.npy"))
featurehog_dataset = np.load(os.path.join(path_saida,"featureHOG_BP.npy"))
featureLBP1_dataset = np.load(os.path.join(path_saida,"featureLBP1_BP.npy"))
featureLBP2_dataset = np.load(os.path.join(path_saida,"featureLBP2_BP.npy"))
classe_image = np.load(os.path.join(path_saida,"classes_BP.npy"))

#DIVIDE DADOS EM TREINAMENTO E TESTE
X_glcm_train, X_glcm_test, y_train, y_test = train_test_split(featureglcm_dataset, classe_image, test_size=0.30, random_state=42)
X_hog_train, X_hog_test, y_train, y_test = train_test_split(featurehog_dataset, classe_image, test_size=0.30, random_state=42)
X_LBP1_train, X_LBP1_test, y_train, y_test = train_test_split(featureLBP1_dataset, classe_image, test_size=0.30, random_state=42)
X_LBP2_train, X_LBP2_test, y_train, y_test = train_test_split(featureLBP2_dataset, classe_image, test_size=0.30, random_state=42)


# TREINANDO OS CLASSIFICADORES PARA TODOS OS EXTRATORES DE CARACTERÍSTICAS
#n_neighbors=3

#knn_GLCM = KNeighborsClassifier(n_neighbors)
#knn_GLCM.fit(X_glcm_train,y_train)

gnb_GLCM = GaussianNB()
gnb_GLCM.fit(X_glcm_train,y_train)

gnb_HOG = GaussianNB()
gnb_HOG.fit(X_hog_train,y_train)

gnb_LBP1 = GaussianNB()
gnb_LBP1.fit(X_LBP1_train,y_train)

gnb_LBP2 = GaussianNB()
gnb_LBP2.fit(X_LBP2_train,y_train)

dataset_path = "images/oilpalm_insight_2_partes" 
image_name = "pedaco_0_0.png"
I = cv2.imread(os.path.join(dataset_path, image_name))

[NL,NC,NCanais] = I.shape

I_Binary_GLCM = np.zeros((NL,NC,NCanais), dtype=int)
I_Binary_HOG = np.zeros((NL,NC,NCanais), dtype=int)
I_Binary_LBP1 = np.zeros((NL,NC,NCanais), dtype=int)
I_Binary_LBP2 = np.zeros((NL,NC,NCanais), dtype=int)

I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

tJan = 47
n_neighbors=3


for i in range(round(tJan/2),NL-round(tJan/2)): 
	for j in range (round(tJan/2),NC-round(tJan/2)):
		print('Classificando pixel na posicao', i, j)
		windowImage = I[i-round(tJan/2):i+round(tJan/2), j-round(tJan/2):j+round(tJan/2)]
		
		# EXTRAI CARACTERISTICAS DE COOCORRENCIA DA JANELA
		featureglcm_window = np.array([]) # guarda as caracteristicas glcm de uma imagem especifica
		glcm = greycomatrix(windowImage, distances=[1,2,3,5,7], angles=[0, np.pi/2], symmetric=True, normed=True)
		props=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
		for prop in props:
			temp = greycoprops(glcm, prop)
			featureglcm_window = np.concatenate((featureglcm_window, temp.flatten()),axis=0)
		
		# EXTRAI CARACTERISTICAS DE HOG DA JANELA
		featurehog_window = hog(windowImage, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
		
		# EXTRAI CARACTERISTICAS LBP DEFAULT (LBP1) DA JANELA
		featureLBP1_window = local_binary_pattern(windowImage, P=24, R=3, method='default')
		
		# EXTRAI CARACTERISTICAS LBP DEFAULT (UNIFORM) DA JANELA
		featureLBP2_window = local_binary_pattern(windowImage, P=24, R=3, method='uniform')

		I_Binary_GLCM[i,j] = gnb_GLCM.predict(featureglcm_window.reshape(1, -1))
		I_Binary_HOG[i,j] = gnb_HOG.predict(featurehog_window.reshape(1, -1))
		I_Binary_LBP1[i,j] = gnb_LBP1.predict(featureLBP1_window.reshape(1, -1))
		I_Binary_LBP2[i,j] = gnb_LBP2.predict(featureLBP2_window.reshape(1, -1))

cv2.imwrite(os.path.join(dataset_path, "resultado_GLCM"+image_name), I_Binary_GLCM)
cv2.imwrite(os.path.join(dataset_path, "resultado_HOG"+image_name), I_Binary_HOG)
cv2.imwrite(os.path.join(dataset_path, "resultado_LBP1"+image_name), I_Binary_LBP1)
cv2.imwrite(os.path.join(dataset_path, "resultado_LBP2"+image_name), I_Binary_LBP2)
		
