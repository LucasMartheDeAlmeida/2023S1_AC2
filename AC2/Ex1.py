import numpy as np
import cv2
import sys
from time import sleep


##primeiramente será necessário esppecificar o caminho do vídeo
VIDEO_CAMINHO = 'C:/Users/Lucas/Documents/visao computacional/2023S1_AC2/videos/cameraEscondida.mp4'


w_min = 5  # largura minima do retangulo
h_min = 5  # altura minima do retangulo
offset = 2  # Erro permitido entre pixel
linha_ROI = 135  # Posição da linha de contagem
movimento = False



def centroide(x, y, w, h):
    """
    :param x: x do objeto
    :param y: y do objeto
    :param w: largura do objeto
    :param h: altura do objeto
    :return: tupla que contém as coordenadas do centro de um objeto
    """
    x1 = w // 2
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy



def Kernel(TIPO_KERNEL):
    if TIPO_KERNEL == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if TIPO_KERNEL == 'opening':
        kernel = np.ones((3,3), np.uint8)
    if TIPO_KERNEL == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel



##agora será necessário definir alguns algorítmos para destacar o fundo
##para compara-los e encontrar o mais coerente para detectar omovimento

tipos_algoritmos = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT','CUSTOM']



Gx = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])
Gy = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])

def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
        return dilation


def Subtractor(tipos_algoritmos):
    if tipos_algoritmos == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = 120, 
                                                        decisionThreshold=0.8)
    if tipos_algoritmos == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history = 100, nmixtures = 5,
                                                        backgroundRatio = 0.7, 
                                                        noiseSigma = 0)
    if tipos_algoritmos == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows=True,
                                                varThreshold=100)
    if tipos_algoritmos == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, 
                                                 detectShadows=True)
    if tipos_algoritmos == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, 
                                                        useHistory =True,
                                                        maxPixelStability=15*60,
                                                        isParallel=True)

 
 
     
    print('Detector inválido')
    sys.exit(1)
    
def Sobel(video,Gx,Gy):
    imgGx = cv2.filter2D(video, ddepth=-1, kernel = Gx)
    imgGy = cv2.filter2D(video, ddepth=-1, kernel = Gy) 
    ##imgTrat =  (imgGy** imgGy + imgGx **imgGx)**(1/2)
    imgTrat =  np.maximum(np.abs(imgGx),np.abs(imgGy))
    
    
    ##aplicando um limiar
    limiarT = 220
    mascara = (
    (imgTrat [:,:,0]<limiarT) &
    (imgTrat [:,:,1]<limiarT) &
    (imgTrat [:,:,2]<limiarT ))
    
    imgFinal =  imgTrat.copy()
    
    mascara =   imgTrat > limiarT
    imgFinal [ mascara ] = 255
    
    mascara =   imgTrat < limiarT
    imgFinal [ mascara ] = 0  
    
    
    
    return imgFinal 

   
w_min = 5  # largura minima do retangulo
h_min = 5  # altura minima do retangulo
offset = 2  # Erro permitido entre pixel
linha_ROI = 135  # Posição da linha de contagem
movimento = False
cap = cv2.VideoCapture(VIDEO_CAMINHO)



def set_info(detec):
    global movimento
    for (x, y) in detec:
        if (linha_ROI + offset) > y > (linha_ROI - offset):
            movimento = True
            cv2.line(frame, (0, linha_ROI), (1200, linha_ROI), (0, 127, 255), 3)
            detec.remove((x, y))
            print("Movimento detectado: " + str(movimento))


detec = []
def set_info(detec):
    global movimento
    for (x, y) in detec:
        if (linha_ROI + offset) > y > (linha_ROI - offset):
            movimento = True
            cv2.line(frame, (0, linha_ROI), (1200, linha_ROI), (0, 127, 255), 3)
            detec.remove((x, y))
            print("Movimento detectado: " + str(movimento))

def mostra_video(mask, video,sobel):
    
    cv2.imshow("mascara", mask)
    cv2.imshow("original", video)
    cv2.imshow("sobel", sobel)

    
tipos_algoritmos = 'GMG'
subtrator_de_fundo = Subtractor(tipos_algoritmos)   




       
while True:

    hasFrame, frame = cap.read() # Pega cada frame do vídeo

    if not hasFrame:
        break



    mask = subtrator_de_fundo.apply(frame)
    mask = Filter(mask, 'combine')
    IMGsobel = Sobel(frame,Gx,Gy) 
    contorno, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (255, 127, 0), 3)
    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= w_min) and (h >= h_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = centroide(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)

    set_info(detec)
    
    mostra_video(mask,frame,IMGsobel)
    
    if cv2.waitKey(1) == 27: #ESC
        break

cv2.destroyAllWindows()
cap.release()