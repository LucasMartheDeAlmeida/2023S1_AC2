import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
CAMINHO_LOGO ='C:/Users/Lucas/Documents/visao computacional/2023S1_AC2/figs/facensLogo.png'

img = cv2.imread('C:/Users/Lucas/Documents/visao computacional/2023S1_AC2/figs/facensLogo.png')
   
rgba_logo= cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    

plt.show(rgba_logo)
plt.show()