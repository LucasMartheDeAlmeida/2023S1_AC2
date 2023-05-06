import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('C:/Users/Lucas/Documents/visao computacional/2023S1_AC2/figs/facensLogo.png')


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])


mask = cv2.inRange(hsv, lower_blue, upper_blue)

result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
result[mask > 0] = [255, 255, 255, 255]
result[mask == 0] = [0, 0, 0, 0]


import cv2
from matplotlib import pyplot as plt
import numpy as np

caminho = 'C:/Users/Lucas/Documents/visao computacional/2023S1_AC2/figs/facensVistaAerea.webp'
img = cv2.imread(caminho)


altura, largura, _ = img.shape
x = largura//2
y = altura//2
cima_esquerda = img[0:y, 0:x]
cima_direita = img[0:y, x:largura]
baixo_esquerda = img[y:altura, 0:x]
baixo_direita = img[y:altura, x:largura]



nova_largura = largura // 2
nova_altura = altura // 2

cima_esquerda_redimensionada = cv2.resize(cima_esquerda, (nova_largura, nova_altura))
cima_direita_redimensionada = cv2.resize(cima_direita, (nova_largura, nova_altura))
baixo_esquerda_redimensionada = cv2.resize(baixo_esquerda, (nova_largura, nova_altura))
baixo_direita_redimensionada = cv2.resize(baixo_direita, (nova_largura, nova_altura))

cima_esquerda_rotacionada = cv2.warpAffine(cima_esquerda_redimensionada, cv2.getRotationMatrix2D((nova_largura/2, nova_altura/2), 45, 1), (nova_largura, nova_altura))
cima_direita_rotacionada = cv2.warpAffine(cima_direita_redimensionada, cv2.getRotationMatrix2D((nova_largura/2, nova_altura/2), -45, 1), (nova_largura, nova_altura))
baixo_esquerda_rotacionada = cv2.warpAffine(baixo_esquerda_redimensionada, cv2.getRotationMatrix2D((nova_largura/2, nova_altura/2), -45, 1), (nova_largura, nova_altura))
baixo_direita_rotacionada = cv2.warpAffine(baixo_direita_redimensionada, cv2.getRotationMatrix2D((nova_largura/2, nova_altura/2), 45, 1), (nova_largura, nova_altura))

nova_imagem = np.zeros((altura, largura, 3), dtype=np.uint8)

nova_imagem[0:nova_altura, 0:nova_largura] = cima_esquerda_rotacionada
nova_imagem[0:nova_altura, nova_largura:largura] = cima_direita_rotacionada
nova_imagem[nova_altura:altura, 0:nova_largura] = baixo_esquerda_rotacionada
nova_imagem[nova_altura:altura, nova_largura:largura] = baixo_direita_rotacionada

nova_imagem_rgb = cv2.cvtColor(nova_imagem, cv2.COLOR_BGR2RGB)




logo = cv2.imread('C:/Users/Lucas/Documents/visao computacional/2023S1_AC2/figs/facensLogo.png')






nova_imagem_rgba = cv2.cvtColor(nova_imagem_rgb, cv2.COLOR_RGB2RGBA)

alturaNovaLogo = int(nova_imagem_rgba.shape[0])
larguraNovaLogo = int(nova_imagem_rgba.shape[1])
canais = int(nova_imagem_rgba.shape[2])


imgResultQuadro = np.zeros((alturaNovaLogo, larguraNovaLogo, canais), dtype=np.uint8)

centro_y = alturaNovaLogo // 2
centro_x = larguraNovaLogo // 2
inicio_y = centro_y - (result.shape[0] // 2)
inicio_x = centro_x - (result.shape[1] // 2)
fim_y = inicio_y + result.shape[0]
fim_x = inicio_x + result.shape[1]
imgResultQuadro[:,:,3] = 0
imgResultQuadro[inicio_y:fim_y, inicio_x:fim_x] = result
alpha_zero = imgResultQuadro[:, :, 3] == 0
imagem_final = np.zeros_like(nova_imagem_rgba)
imagem_final = np.where(alpha_zero[..., np.newaxis], nova_imagem_rgba, imagem_final)
imagem_final = np.where(alpha_zero[..., np.newaxis] == False, imgResultQuadro, imagem_final)
imagem_final = cv2.cvtColor(imagem_final, cv2.COLOR_RGBA2RGB)


plt.imshow(imagem_final)
plt.show()



