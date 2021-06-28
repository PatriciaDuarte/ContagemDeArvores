import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

J = cv2.imread("images/oilpalm-insight-2.jpg")
print("Imagem lida.")
nLinhas = J.shape[0]
nColunas = J.shape[1]
N = nLinhas//8
M = nColunas//8

for x in range(0,nLinhas,N):
    for y in range(0, nColunas, M):
        cv2.imwrite("images/oilpalm_insight_2_partes/pedaco_"+str(x)+"_"+str(y)+".png",J[x:x+N, y:y+M,:])