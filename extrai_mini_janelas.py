import cv2
import numpy as np


J = cv2.imread("images/oilpalm_insight_2_partes/pedaco_13916_40170.png")
I = np.copy(J)
a = []
b = []
tJan = 47

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.rectangle(J, (x-round(tJan/2), y-round(tJan/2)), (x+round(tJan/2), y+round(tJan/2)), (0, 0, 255), thickness=2)
        cv2.putText(J, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", J)
        cv2.imwrite("images/oilpalm_insight_2_partes/pedaco_0_0/janelas_fora_do_centro/pedaco"+str(x)+"_"+str(y)+".png", I[y-round(tJan/2):y+round(tJan/2), x-round(tJan/2):x+round(tJan/2), :])
        print(x,y)

while(1):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", J)
    cv2.waitKey(0)
