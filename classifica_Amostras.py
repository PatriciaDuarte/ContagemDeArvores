import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


path_saida = "dataset_features"


featureglcm_dataset = np.load(os.path.join(path_saida,"featureGLCM.npy"))
featurehog_dataset = np.load(os.path.join(path_saida,"featureHOG.npy"))
featureLBP1_dataset = np.load(os.path.join(path_saida,"featureLBP1.npy"))
featureLBP2_dataset = np.load(os.path.join(path_saida,"featureLBP2_dataset.npy"))
classe_image = np.load(os.path.join(path_saida,"classes.npy"))

# Exemplo de como concatenar múltiplas características
featureglcm_hog_dataset = np.concatenate((featureglcm_dataset, featurehog_dataset), axis=1)



n_neighbors=3


# Classificador kNN usando GLCM
X_glcm_train, X_glcm_test, y_train, y_test = train_test_split(featureglcm_dataset, classe_image, test_size=0.30, random_state=42)
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_glcm_train,y_train)
y_pred = knn.predict(X_glcm_test)
print("Acuracia usando GLCM e kNN com k=",n_neighbors,":",metrics.accuracy_score(y_test, y_pred))
print("\n")




#################################################################################################
# Classificador kNN usando HOG e kNN
X_hog_train, X_hog_test, y_train, y_test = train_test_split(featurehog_dataset, classe_image, test_size=0.30, random_state=42)
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_hog_train,y_train)
y_pred = knn.predict(X_hog_test)
print("Acuracia usando HOG e kNN com k=",n_neighbors,":",metrics.accuracy_score(y_test, y_pred))
print("\n")




#################################################################################################
# Classificador kNN usando LBP1
X_LBP1_train, X_LBP1_test, y_train, y_test = train_test_split(featureLBP1_dataset, classe_image, test_size=0.30, random_state=42)
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_LBP1_train,y_train)
y_pred = knn.predict(X_LBP1_test)
print("Acuracia usando LBP1 e kNN com k=",n_neighbors,":",metrics.accuracy_score(y_test, y_pred))
print("\n")




#################################################################################################
# Classificador kNN usando LBP1
X_LBP2_train, X_LBP2_test, y_train, y_test = train_test_split(featureLBP2_dataset, classe_image, test_size=0.30, random_state=42)
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_LBP2_train,y_train)
y_pred = knn.predict(X_LBP2_test)
print("Acuracia usando LBP2 e kNN com k=",n_neighbors,":",metrics.accuracy_score(y_test, y_pred))
print("\n")



#################################################################################################
# Classificador kNN usando GLCM+HOG
X_GLCM_HOG_train, X_GLCM_HOG_test, y_train, y_test = train_test_split(featureglcm_hog_dataset, classe_image, test_size=0.30, random_state=42)
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_GLCM_HOG_train,y_train)
y_pred = knn.predict(X_GLCM_HOG_test)
print("Acuracia usando GLCM+HOG e kNN com k=",n_neighbors,":",metrics.accuracy_score(y_test, y_pred))
print("\n")









