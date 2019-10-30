import cv2
import numpy as np

img = cv2.imread('manga1.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(th1,kernel,iterations = 1)
# Escreve o nome do filtro na imagem correspondente
font = cv2.FONT_HERSHEY_SIMPLEX
imagemFinal = cv2.putText(dilation,'',(45,320), font, 1, (0,0,255), 2, cv2.LINE_AA)

# Salva a imagem
cv2.imwrite("dilatacao.png", imagemFinal)
# Mostra a imagem final concatenada
cv2.imshow("Imagem final:", imagemFinal)
# Aguarda tecla para finalizar
cv2.waitKey(0)