#Importa as bibliotecas que serao usadas no programa
import cv2
import numpy as np

#Pega a imagem que sera processada
imagem = cv2.imread('imagem.jpg',0)

#aplica o thresholding metodo de otsu
ret,thresholding = cv2.threshold(imagem,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Escreve o nome do filtro na imagem correspondente
font = cv2.FONT_HERSHEY_SIMPLEX
imagemFinal = cv2.putText(thresholding,'Limiarizacao usando metodo de otsu',(45,320), font, 1, (0,0,255), 2, cv2.LINE_AA)

# Salva a imagem
cv2.imwrite("imagemFinal.png", imagemFinal)
# Mostra a imagem final concatenada
cv2.imshow("Imagem final:", imagemFinal)
# Aguarda tecla para finalizar
cv2.waitKey(0)
