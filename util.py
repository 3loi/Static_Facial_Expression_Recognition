
from tqdm import tqdm
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np


net = cv2.dnn.readNetFromCaffe('/media/a/storage/util/face-alignment/deploy.prototxt.txt','/media/a/storage/util/face-alignment/res10_300x300_ssd_iter_140000.caffemodel') 

def make_square(startX, startY, endX, endY, width, height):
    lenx = endX - startX
    leny = endY - startY
    if leny > lenx:
        diff = (leny - lenx)/2
        startX -= diff
        endX += diff
        
    elif lenx > leny:
        diff = (lenx - leny)/2
        startY -= diff
        endY += diff
        
    startX = startX if startX > 0 else 0
    endX = endX if endX < width else width
    startY = startY if startY > 0 else 0
    endY = endY if endY < height else height
    
    return int(startX), int(startY), int(endX), int(endY)


#takes image as input, resizes it and runs it through a face detector. 
#returns one face at a time
face_locations = {}
def my_face_detect(path, image, conf = 0.5, width = 300, height = 300):
    #print(type(image))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    faces = []
    i = 0
    while True:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY, endX, endY) = make_square(startX, startY, endX, endY, w, h)
        face_locations[path] = (startX, startY, endX, endY)
    #             print('face box',startX, startY, endX, endY)
    #             print(box)
        face = image[startY:endY,startX:endX]
        if face.shape[0] != 0 and face.shape[1] != 0:
            break
        i += 1
        if detections.shape[2] >= i:
            return
    yield face
    return


def face_get(image):
    img = cv2.imread(image)
    if img is None:
        return
    faces = my_face_detect(image, img)
    for i, d in enumerate(faces):
        #save image if needed
        return d
    return 
