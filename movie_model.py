import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
#Functions:
# import diverse_programs.Marian.no_onions as y_func
# import diverse_programs.Glen_Roger.rotate_mirror_image as rotate_func
import diverse_programs.Jeanette.model_functions as model_func
# import diverse_programs.Jeanette.picture_functions as pict_func

import time

#Setting up the modell

MODEL_NAME      = "ResNet50"
# LOAD_MODEL_NAME = "04-resnet50"
LOAD_MODEL_NAME = "7-resnet50"
SCALE_WIDTH     = 256
SCALE_HEIGHT    = 192
baseModel = model_func.get_model(MODEL_NAME, SCALE_WIDTH, SCALE_HEIGHT)

for layer in baseModel.layers:
    layer.trainable = False

out = baseModel.layers[-1].output
output = tf.keras.layers.Dense(1, activation='sigmoid')(out)

model = tf.keras.models.Model(inputs=baseModel.input, outputs=output)
# model.summary()

load_filepath = f"./TEMP_MODEL_WEIGHTS/{LOAD_MODEL_NAME}/{LOAD_MODEL_NAME}"
model.load_weights(load_filepath)


#Setting up the camera
cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture(0)
print("Go!")


#For FPS
start_time = time.time()
x = 1
counter = 0
fps = 0

while True:

        

    check, frame = cam.read()

    # frame[:,:,0] -= np.uint8(frame[:,:,0]/2) - 40
    # frame[:,:,1] += 10
    # frame[:,:,2] -= np.uint8(frame[:,:,2]/2) - 40

    # res = cv2.resize(frame, dsize=(SCALE_WIDTH, SCALE_HEIGHT), interpolation=cv2.INTER_CUBIC)

    res = cv2.resize(frame, fx = SCALE_WIDTH, fy = SCALE_HEIGHT, dsize=(SCALE_WIDTH, SCALE_HEIGHT), interpolation=cv2.INTER_CUBIC)


    X_list = [res]
    X_tmp = np.array(X_list)

    X = np.array(X_tmp)
    X = X/255

    X = model_func.get_preprocess_input(MODEL_NAME, X)

    pred_test = model.predict(X, verbose=0)



    pic_to_show = frame


    fo_warning = "Everything looks okay!"

    # if pred_test[0][0] > .45:
    #     fo_warning = "Foreign object detected!"
    #     pic_to_show[0:200][0:139][0]      = 0
    #     pic_to_show[0:200][0:139][1]      = 0
    #     pic_to_show[0:200][0:139][2]      = 0
    #     pic_to_show[0:20][0:19][0]      = 0
    #     pic_to_show[0:20][0:19][1]      = 0
    #     pic_to_show[0:20][0:19][2]      = 0        

    cv2.putText(img=pic_to_show, text=fo_warning,org=(10,20), fontFace=1, fontScale=1.5, color=(0,0,120), thickness=2)
    # cv2.putText(img=pic_to_show, text=f"Prediction: {pred_test[0][0]}",org=(10,40), fontFace=1, fontScale=1, color=(120,0,0), thickness=2)
    cv2.putText(img=pic_to_show, text=f"Prediction: {int(pred_test[0][0]*1000)/10}%",org=(10,40), fontFace=1, fontScale=1, color=(120,0,0), thickness=2)
    cv2.putText(img=pic_to_show, text=f"FPS: {fps}",org=(10,60), fontFace=1, fontScale=1, color=(120,0,0), thickness=2)



    # cv2.putText(img=frame, text="Hei",org=(100,100), fontFace=2, fontScale=8, color=(255,18,18))
    # cv2.addText(img=frame,text="HEi", org=(10,10), nameFont="arial.tff")
    cv2.imshow('video', pic_to_show)

    key = cv2.waitKey(1)
    if key == 27:  #Esc
        break

    counter += 1


    if time.time() - start_time > x:
        fps = counter
        # print(f"FPS: {counter}")
        counter = 0
        start_time = time.time()

cam.release()
cv2.destroyAllWindows()




#Deleting..
# for i in range(100):
#     print(i,end="")
#     print("\b",end="")