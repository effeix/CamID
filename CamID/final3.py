# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import cv
import PyQt4.QtGui as QtGui
import PyQt4.QtCore as QtCore
from time import sleep

class CardReader(QtGui.QWidget):
    def __init__(self):
        super(CardReader, self).__init__()
        
        #SETTINGS
        self.setWindowTitle("Churca CardReader")
        self.setWindowIcon(QtGui.QIcon("favicon.png"))
        self.setFixedSize(230,150)
        self.move(70,70)
        
        self.header = QtGui.QLabel("Escolha a empresa:")
                
        
        #LAYOUT
        gridLayout = QtGui.QGridLayout(self)
        gridLayout.addWidget(Empresa(iconPath="insper.png",img="insper.jpg",name="Insper"),1,0)
        gridLayout.addWidget(Empresa(iconPath="espm.png",img="espm.jpg",name="ESPM"),1,1)
        gridLayout.addWidget(self.header,0,0)        
        
        self.show()
        
class Empresa(QtGui.QLabel):
    def __init__(self, iconPath="", img="", name=""):
        super(Empresa, self).__init__()
        
        #VARIABLES
        self._iconPath = iconPath
        self._name     = name
        self._img      = img
        self._pixmap = QtGui.QPixmap(self._iconPath)
        
        #SETTINGS
        self.setToolTip(self._name)
        self.mouseReleaseEvent = self.ler_carteira
        self.setPixmap(self._pixmap)
    
    #HOVER AND CLICK EVENTS    
    def enterEvent(self, event):
        self.resize(self.width(), self.height()+10)
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def leaveEvent(self, event):
        self.resize(self.width(), self.height()-10)
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
    
    #RUN OPENCV    
    def ler_carteira(self, e):
        ESC=27 #Código ASCII para a tecla 'ESC'  
        camera = cv2.VideoCapture(1) #Pega o feed da câmera (0 é a câmera embutida)
        orb = cv2.ORB() #Cria uma instância ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #Usa o bruteforce keypoint descriptor com o método hamming
        
        imgoriginalColor=cv2.imread(self._img) #Lê a imagem modelo
        imgoriginalGray = cv2.cvtColor(imgoriginalColor, cv2.COLOR_BGR2GRAY) #Converte imagem modelo em formato BGR para Escala de Cinza
        
        kpTrain = orb.detect(imgoriginalGray,None) #Encontra os keypoints da imagem original
        kpTrain, desTrain = orb.compute(imgoriginalGray, kpTrain) #Computa todos os descriptors dos keypoints detectados
        
        firsttime=True
        
        while True:
           
            ret, imgCamColor = camera.read() #lê as informações da câmera
            imgCamGray = cv2.cvtColor(imgCamColor, cv2.COLOR_BGR2GRAY) #Converte imagem da câmera em Escala de Cinza
            kpCam = orb.detect(imgCamGray,None) #Enontra os keypoints da imagem da câmera
            kpCam, desCam = orb.compute(imgCamGray, kpCam) #Computa todos os descriptors dos keypoints detectados
            matches = bf.match(desCam,desTrain) #Aplica o matching entre os keypoints das duas imagens
            dist = [m.distance for m in matches] #Coloca um threshold entre a distância dos arrays para detectar os erros 
            thres_dist = (sum(dist) / len(dist)) * 0.5
            matches = [m for m in matches if m.distance < thres_dist] #Elimina os erros baseados no threshold acima
        
            if firsttime==True: #Coloca um tamanho para as imagens sendo comparadas para depois serem mostradas para o usuário
                h1, w1 = imgCamColor.shape[:2]
                h2, w2 = imgoriginalColor.shape[:2]
                nWidth = w1+w2 
                nHeight = max(h1, h2)
                hdif = (h1-h2)/2
                firsttime=False
               
            result = np.zeros((nHeight, nWidth, 3), np.uint8)
            result[hdif:hdif+h2, :w2] = imgoriginalColor #Pega os resultados das duas imagens em cores
            result[:h1, w2:w1+w2] = imgCamColor
        
            for i in range(len(matches)):
                pt_a=(int(kpTrain[matches[i].trainIdx].pt[0]), int(kpTrain[matches[i].trainIdx].pt[1]+hdif)) #Cria linhas pra cada matching point achado pelo programa 
                pt_b=(int(kpCam[matches[i].queryIdx].pt[0]+w2), int(kpCam[matches[i].queryIdx].pt[1]))
                cv2.line(result, pt_a, pt_b, (255, 0, 0))
        
            cv2.imshow('Câmera', result) #Mostra a imagem da câmera e da imagem original
            print(len(matches)) #Imprime o número de matches entre a câmera e a imagem original
            if len(matches)>14: #Checa se a cartão verificado é o correto pelo número de matches
                accdec = AcceptDecline(msg="Desconto concedido!")
                break
            else:
                accdec = AcceptDecline(msg="Tente novamente! Ou aperte ESC para sair")
           
            
            key = cv2.waitKey(20)                             
            if key == ESC: #Espera a tecla esc ser apertada para fechar o programa
                break
        
        cv2.destroyAllWindows()
        camera.release()
        cardreader.close()
        
class AcceptDecline(QtGui.QWidget):
    def __init__(self, msg=""):
        super(AcceptDecline, self).__init__()
        
        #SETTINGS
        self.setWindowTitle("Churca CardReader")
        self.setWindowIcon(QtGui.QIcon("favicon.png"))
        self.setFixedSize(350,100)
        self.move(70,70)
        
        self._msg = msg
        self.labelAD = QtGui.QLabel(self._msg)
        
        #LAYOUT
        gridLayout = QtGui.QGridLayout(self)
        gridLayout.addWidget(self.labelAD,0,0)
        
        if self._msg == "Desconto concedido!":
            self.show()
            sleep(4)
            QtGui.QApplication.quit()
        else:
            self.show()
            
        
#ler_carteira()
        
app = QtGui.QApplication(sys.argv)
cardreader = CardReader()
sys.exit(app.exec_())