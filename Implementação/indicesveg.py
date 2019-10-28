import os, glob
import numpy as np
from skimage import filters
from scipy.misc import imread, imsave
from scipy import ndimage as ndi


from matplotlib.colors import rgb_to_hsv
from skimage.color import rgb2gray
from skimage.morphology import disk, opening, watershed
from skimage.feature import peak_local_max
from random import random
import matplotlib
import matplotlib.pyplot as plt


def normalize0_255(X):
    X = (X - X.min()) / (X.max() - X.min()) * 255
    return X.round()

def matrixDivision(M, N):
    C = np.divide(M, N, out=np.zeros_like(M), where=N != 0)
    return C

def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return v

dataset_path = "images"
results_output = "images_results"
image_types = ['jpg', 'JPG', 'png']
#for f in glob.glob(".*"):
#    os.remove(f)
file_names = [fn for fn in os.listdir(dataset_path)
              if any(fn.endswith(ext) for ext in image_types)]

colors = [(1,1,1)] + [(random(),random(),random()) for i in range(2048)]


for file in file_names:
    print("Processing file: ", file)
    L = list()

    J = imread(os.path.join(dataset_path,file))
    L.append(J)

    I = J/255
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]
    RGB = R+G+B

    r = matrixDivision(R, RGB)
    g = matrixDivision(G, RGB)
    b = matrixDivision(B, RGB)

    # Eq (1)
    #NGRDI = r
    #NGRDI = matrixDivision(G-R, G+R)
    #Indice = 0.481*g + 0.45*b #- 0.841*r
    
    HSV = rgb_to_hsv(I)
    
    Indice =  HSV[:,:,2]
    
    Indice_norm = normalize0_255(Indice)
    L.append(Indice_norm)
    
    thr = filters.threshold_otsu(Indice_norm)
    Indice_bin =  Indice_norm < thr - 10
    newIndice_bin = np.vectorize(boolstr_to_floatstr)(Indice_bin).astype(float)
    L.append(newIndice_bin)

    Indice_binOpened = opening(Indice_bin, disk(2))  # tentei com disk(3) mais destroi muito
    newIndice_binOpened = np.vectorize(boolstr_to_floatstr)(Indice_binOpened).astype(float)
    L.append(newIndice_binOpened)

    #PATRICIA: Estudar a função distance_transform_edt (Transformada da Distância) e escrever sobre ela
    #Atualizado: Função que separa  dois objetos na imagem, calculando a distância dos pontos diferentes de zero
    # (ou seja, sem fundo) até o ponto mais próximo de zero (ou seja, fundo).
    #Retorna  matriz de índices
    distanceTranf = ndi.distance_transform_edt(newIndice_binOpened)
    

    #PATRICIA: Estudar a função peak_local_max e seus parâmetros, e escrever sobre detecção de picos
   # Atualizado:Gerando os marcadores como máximos locais da distância do plano de fundo
   #A peak_local_maxfunção retorna as coordenadas dos picos locais (máximos) em uma imagem. Um filtro 
   #máximo é usado para encontrar os máximos locais. Esta operação dilata a imagem original e mescla os 
   #máximos locais vizinhos mais perto do que o tamanho da dilatação. Os locais onde a imagem original é
   # igual à imagem dilatada são retornados como máximos locais.
    local_maxi = peak_local_max(distanceTranf, indices=False, footprint=np.ones((3, 3)),
                            labels=newIndice_binOpened)
    L.append(distanceTranf)

    markers = ndi.label(local_maxi)[0]
    #PATRICIA: Estudar a função watershed e seus parâmetros
    labels = watershed(-distanceTranf, markers, mask=newIndice_binOpened)
    L.append(labels)

    # PATRICIA: Descrever índices de vegetação
    
    # Eq (2)
    # ExG = g
    # #ExG = 1.5* matrixDivision((g-r), (g+r+0.5))
    # #ExG = 2 *r + 1 * g + 1 * b
    # ExG_norm = normalize0_255(ExG)
    # L.append(ExG_norm)
    # thr = filters.threshold_otsu(ExG_norm)
    # ExG_bin = ExG_norm < thr
    # newExG_bin = np.vectorize(boolstr_to_floatstr)(ExG_bin).astype(float)
    # L.append(newExG_bin)


    # # Eq (3)
    # #CIVE = 0.441*r - 0.881*g + 0.385*b + 18.78745
    # CIVE = b
    # #CIVE = b+r - g
    # CIVE_norm = normalize0_255(CIVE)
    # L.append(CIVE_norm)
    # thr = filters.threshold_otsu(CIVE_norm)
    # CIVE_bin = CIVE_norm < thr
    # newCIVE_bin = np.vectorize(boolstr_to_floatstr)(CIVE_bin).astype(float)
    # L.append(newCIVE_bin)

    # # Eq (4)
    # a=0.667
    # #VEG = matrixDivision(g,np.power(r,a)*np.power(b,1-a))
    # VEG = 2*r - g - b
    # VEG_norm = normalize0_255(VEG)
    # L.append(VEG_norm)
    # thr = filters.threshold_otsu(VEG_norm)
    # VEG_bin = VEG_norm < thr
    # newVEG_bin = np.vectorize(boolstr_to_floatstr)(VEG_bin).astype(float)
    # L.append(newVEG_bin)

    # # Eq (5)
    # ExGR = ExG -1.4*r - g
    # ExGR_norm = normalize0_255(ExGR)
    # L.append(ExGR_norm)
    # thr = filters.threshold_otsu(ExGR_norm)
    # ExGR_bin = ExGR_norm < thr
    # newExGR_bin = np.vectorize(boolstr_to_floatstr)(ExGR_bin).astype(float)
    # L.append(newExGR_bin)

    # # Eq (6)
    # WI = matrixDivision(g-b, r-g)
    # WI_norm = normalize0_255(WI)
    # L.append(WI_norm)
    # thr = filters.threshold_otsu(WI_norm)
    # WI_bin = WI_norm < thr
    # newWI_bin = np.vectorize(boolstr_to_floatstr)(WI_bin).astype(float)
    # L.append(newWI_bin)

    # # Eq (7)
    # COM = 0.25*ExG + 0.3*ExGR + 0.33*CIVE + 0.12*VEG
    # COM_norm = normalize0_255(COM)
    # L.append(COM_norm)
    # thr = filters.threshold_otsu(COM_norm)
    # COM_bin = COM_norm < thr
    # newCOM_bin = np.vectorize(boolstr_to_floatstr)(COM_bin).astype(float)
    # L.append(newCOM_bin)

    # # Eq (8)
    # COM2 = 0.36*ExG + 0.47*CIVE + 0.17*VEG
    # COM2_norm = normalize0_255(COM2)
    # L.append(COM2_norm)
    # thr = filters.threshold_otsu(COM2_norm)
    # COM2_bin = COM2_norm < thr
    # newCOM2_bin = np.vectorize(boolstr_to_floatstr)(COM2_bin).astype(float)
    # L.append(newCOM2_bin)

    methodsName = ['00ORIG', '01Indice', '02Indice_bin', '03Indice_binOpened', '04distanceTranf', '05labels']

    for i,IMG in enumerate(L):
        filename, file_ext = os.path.splitext(file)
        imsave(results_output+'/'+filename+'_'+methodsName[i]+'.png', IMG, 'png')
        new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=(np.unique(labels)).size)
        imsave(results_output+'/'+filename+'_05labels.png', new_map(labels))      

