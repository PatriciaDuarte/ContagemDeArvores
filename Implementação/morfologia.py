#importa as bibliotecas que serao utilizadas
import cv2
import numpy as np

#Pega a imagem que sera processada
imagem = cv2.imread('bird.jpg')

# Aplica os respectivos filtros
kernel = np.ones((6,6),np.float32)/25
filtro2D = cv2.filter2D(imagem,-1,kernel)
borrao = cv2.blur(imagem,(5,5))
borraoGaussiano = cv2.GaussianBlur(imagem,(5,5),0)
mediana = cv2.medianBlur(imagem,5)
filtragemBilateral = cv2.bilateralFilter(imagem,9,75,75)

# Escreve o nome de cada filtro na imagem correspondente
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imagem,'Imagem original',(5,320), font, 1, (0,0,255), 2, cv2.LINE_AA)
cv2.putText(filtro2D,'Filtro 2D',(45,320), font, 1, (0,0,255), 2, cv2.LINE_AA)
cv2.putText(borrao,'Borrao',(80,320), font, 1, (0,0,255), 2, cv2.LINE_AA)
cv2.putText(borraoGaussiano,'BorraoGaussiano',(10,320), font, 1, (0,0,255), 2, cv2.LINE_AA)
cv2.putText(mediana,'Mediana',(60,320), font, 1, (0,0,255), 2, cv2.LINE_AA)
cv2.putText(filtragemBilateral,'Filtro Bilateral',(10,320), font, 1, (0,0,255), 2, cv2.LINE_AA)

# Concatena as imagens para gerar a imagem de saida
imagemTemporaria1 = np.concatenate((imagem, mediana, borraoGaussiano), axis=1)
imagemTemporaria2 = np.concatenate((filtragemBilateral, borrao, filtro2D), axis=1)
imagemFinal = np.concatenate((imagemTemporaria1, imagemTemporaria2), axis=0)

cv2.imwrite("bird1.png", imagemFinal)
cv2.imshow("Imagem Final:", imagemFinal)
cv2.waitKey(0)