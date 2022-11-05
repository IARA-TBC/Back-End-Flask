import cv2
import os
import torch
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       'best.pt')  # custom trained model
import io
from PIL import Image
import base64

def image_TBC_location(im):
    results = model(im)
    # Results
    # results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    # results.save()  # or .show(), .save(), .crop(), .pandas(), etc.

    #results.save()
    coordenadas = results.xyxy[0].numpy()  # im predictions (tensor)

    return coordenadas

def new_img_with_coordenates(cor, old_path, name):
    if(len(cor) == 2):  
        img = cv2.imread(old_path)
        print(cor[0][0])
        cv2.rectangle(img, (int(cor[0][0]), int(cor[0][1])), (int(cor[0][2]), int(cor[0][3])), (255, 0, 0), 3)
        cv2.rectangle(img, (int(cor[1][0]), int(cor[1][1])), (int(cor[1][2]), int(cor[1][3])), (255, 0, 0), 3)
        path = os.getcwd() + '/images'
        path = path + '/' + name
        cv2.imwrite(path, img)
        return path
    elif(len(cor) == 3):  
        img = cv2.imread(old_path)
        print(cor[0][0])
        cv2.rectangle(img, (int(cor[0][0]), int(cor[0][1])), (int(cor[0][2]), int(cor[0][3])), (255, 0, 0), 3)
        cv2.rectangle(img, (int(cor[1][0]), int(cor[1][1])), (int(cor[1][2]), int(cor[1][3])), (255, 0, 0), 3)
        cv2.rectangle(img, (int(cor[2][0]), int(cor[2][1])), (int(cor[2][2]), int(cor[2][3])), (255, 0, 0), 3)
        path = os.getcwd() + '/images'
        path = path + '/' + name
        cv2.imwrite(path, img)
        return path
    elif(len(cor) == 1):
        img = cv2.imread(old_path)
        print(cor)
        cv2.rectangle(img, (int(cor[0][0]), int(cor[0][1])), (int(cor[0][2]), int(cor[0][3])), (255, 0, 0), 3)
        path = os.getcwd() + '/images'
        path = path + '/' + name
        cv2.imwrite(path, img)
        return path
    else:
        return old_path


