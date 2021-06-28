import os
import csv
import math
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops, hog, local_binary_pattern
import matplotlib.pyplot as plt

dataset_path = "textura_recorte"
classes = ["centro", "naocentro"]
path_saida = "dataset_features"

image_types = ['jpg', 'JPG', 'png']


index = 0
classe_image = [] # 0 = naocentro; 1 = centro
for classe in classes:
	file_names = [fn for fn in os.listdir(os.path.join(dataset_path,classe)) if any(fn.endswith(ext) for ext in image_types)]
	for file in file_names:
		print("Processando arquivo ",file, " da classe", classe)
		filename, file_extension = os.path.splitext(file)

		J = cv2.imread(os.path.join(dataset_path, classe, file))
		grayImage = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
		
		
		# EXTRAI CARACTERISTICAS DE COOCORRENCIA
		featureglcm_image = np.array([]) # guarda as caracteristicas glcm de uma imagem especifica
		glcm = greycomatrix(grayImage, distances=[1,2,3,5,7], angles=[0, np.pi/2], symmetric=True, normed=True)
		props=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
		
		for prop in props:
			temp = greycoprops(glcm, prop)
			featureglcm_image = np.concatenate((featureglcm_image, temp.flatten()),axis=0)
		if index==0:
			featureglcm_dataset = np.copy(featureglcm_image)
			featureglcm_dataset = np.reshape(featureglcm_dataset,(1, featureglcm_dataset.size))

		else:
			featureglcm_dataset = np.append(featureglcm_dataset, np.array([featureglcm_image.T]),axis=0)
		
		# EXTRAI CARACTERISTICAS DE HOG
		featurehog_image = hog(grayImage, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
		if index==0:
			featurehog_dataset = np.copy(featurehog_image)
			featurehog_dataset = np.reshape(featurehog_dataset,(1, featurehog_dataset.size))
		else:
			featurehog_dataset = np.append(featurehog_dataset, np.array([featurehog_image.T]),axis=0)

		# EXTRAI CARACTERISTICAS LBP DEFAULT (LBP1)
		featureLBP1_image = local_binary_pattern(grayImage, P=24, R=3, method='default')
		if index==0:
			featureLBP1_dataset = np.copy(featureLBP1_image)
			featureLBP1_dataset = np.reshape(featureLBP1_dataset,(1, featureLBP1_dataset.size))
		else:
			featureLBP1_image = np.reshape(featureLBP1_image,(1, featureLBP1_image.size))
			featureLBP1_dataset = np.append(featureLBP1_dataset, featureLBP1_image,axis=0)

		# EXTRAI CARACTERISTICAS LBP DEFAULT (UNIFORM)
		featureLBP2_image = local_binary_pattern(grayImage, P=24, R=3, method='uniform')
		if index==0:
			featureLBP2_dataset = np.copy(featureLBP2_image)
			featureLBP2_dataset = np.reshape(featureLBP2_dataset,(1, featureLBP2_dataset.size))
		else:
			featureLBP2_image = np.reshape(featureLBP2_image,(1, featureLBP2_image.size))
			featureLBP2_dataset = np.append(featureLBP2_dataset, featureLBP2_image,axis=0)
		if classe == "naocentro":
			classe_image.append(0)
		else:
			classe_image.append(1)

		index = index + 1

# SALVA ARQUIVOS PARA USO POSTERIOR
print("Salvando as caracteristicas ...")
np.save(os.path.join(path_saida,"featureGLCM.npy"),featureglcm_dataset)
np.save(os.path.join(path_saida,"featureHOG.npy"),featurehog_dataset)
np.save(os.path.join(path_saida,"featureLBP1.npy"),featureLBP1_dataset)
np.save(os.path.join(path_saida,"featureLBP2_dataset.npy"),featureLBP2_dataset)
np.save(os.path.join(path_saida,"classes.npy"),np.array(classe_image))
print("Caracteristicas salvas. FIM")