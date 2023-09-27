import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
#Functions:
import diverse_programs.Marian.no_onions as y_func
import diverse_programs.Glen_Roger.rotate_mirror_image as rotate_func
import diverse_programs.Jeanette.model_functions as model_func
import diverse_programs.Jeanette.picture_functions as pict_func

MODEL_NAME = 'efficientnetv2-xl-21k' # Xception, ResNet50, InceptionResNetV2, DenseNet169
VERSION = '01'
USER = 'JG'
SCALE_WIDTH = 256
SCALE_HIGHT = 192

FIT_MODEL = True #if false - load model
SAVE_MODEL = True
SAVE_MODEL_NAME = f'01_{MODEL_NAME}' #MÅ OPPDATERES HVER GANG.
if SAVE_MODEL_NAME+'.index' in os.listdir(f'TEMP_MODEL_WEIGHTS/'):
    input_answer = input("This model already exists. Please change SAVE_MODEL_NAME. Do you want to overwrite? y/n: ")
    if input_answer.lower() != 'y':
        exit(0)
#IF FIT_MODEL = FALSE - USE LOAD_MODEL_NAME:
LOAD_MODEL_NAME = ''

IMAGE_PATHS = ["Bilder/256x192/Green/", "Bilder/256x192/Normal/"]

#MODEL
baseModel = model_func.get_model(MODEL_NAME, SCALE_WIDTH, SCALE_HIGHT)

#PICTURES TO MODEL
pictures = pict_func.get_pictures(IMAGE_PATHS)
picture_names = pict_func.get_picture_names(IMAGE_PATHS)
picture_names_with_path = pict_func.get_picture_names_with_path(IMAGE_PATHS)

#PREPROCESS X:
X = np.array(pictures)
X = X/255

X = model_func.get_preprocess_input(MODEL_NAME, X)

#CREATE y:
y_list = y_func.no_onions_function(picture_names) #[0,1,0,1,1,0]
y = np.array(y_list)
len(y) #166


y_train_with_index = np.c_[y, range(len(y))]
# print(y_train_with_index[27])
# type(y_train_with_index)
# y_train_with_index.shape


#split data
X_train, X_test, y_train, y_test = train_test_split(X, y_train_with_index, test_size=0.2, random_state=42)
X_train.shape #(132, 192, 256, 3)
X_test.shape #(34, 192, 256, 3)
y_train.shape #(132,)
y_test.shape #(34,)

y_train_index = y_train[:,1]
y_train = y_train[:,0]
y_test_index = y_test[:,1]
y_test = y_test[:,0]
# print(y_train)
# print(y_train_index)

#expand X_train
#rotate_image(picture_names_with_path, y_list)

# print(dir(model))
# model._output_layers
baseModel.summary()

for layer in baseModel.layers:
    layer.trainable = False

out = baseModel.layers[-1].output
output = tf.keras.layers.Dense(1, activation='sigmoid')(out)

model = tf.keras.models.Model(inputs=baseModel.input, outputs=output)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

if SAVE_MODEL:
    save_filepath = f"./TEMP_MODEL_WEIGHTS/{SAVE_MODEL_NAME}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max')
import time
print(y_train.shape)
# if 'Value error: 'logits' and 'labels' not the same shape...check model_functions: pooling=max.
if FIT_MODEL:
    st = time.process_time()
    history = model.fit(X_train, y_train,
        validation_data=[X_test,y_test],
        batch_size=32,
        epochs=5,
        callbacks=[model_checkpoint_callback])
    et = time.process_time()
    elapsed_time = et - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    model_process_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
else:
    load_filepath = f"./TEMP_MODEL_WEIGHTS/{LOAD_MODEL_NAME}"
    model.load_weights(load_filepath)


history.history.keys()
history.history['loss']
history.history['val_loss']

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper left')
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.legend(loc='upper left')
plt.show()


#TRAIN SCORE
pred_train = model.predict(X_train)
y_val_pred_class = np.where(pred_train > 0.5, 1, 0)
y_val_pred_class.shape
score_train = accuracy_score(y_train, y_val_pred_class)
print(f"Score train: {score_train}")

cm_train = confusion_matrix(y_train,y_val_pred_class)
ConfusionMatrixDisplay(cm_train).plot()
plt.show()


#TEST SCORE
pred_test = model.predict(X_test)
y_val_pred_test = np.where(pred_test > 0.5, 1, 0)
y_val_pred_test.shape
score_test = accuracy_score(y_test, y_val_pred_test)
print(f"Score test: {score_test}")

cm_test = confusion_matrix(y_test,y_val_pred_test)
ConfusionMatrixDisplay(cm_test).plot()
plt.show()

tn = cm_test[0,0]
fp = cm_test[0,1]
fn = cm_test[1,0]
tp = cm_test[1,1]

precision = tp / (tp + fp) #0.9123434704830053
#av alle modellen sier er true, så treffer den med 91%

recall = tp / (tp + fn) #0.8252427184466019
#av alle true som finnes, så treffer modellen 83%

fp_cost = fp * 5 #245 #FP = modellen sier det er er stein på båndet, men det er det ikke.
fn_cost = fn * 100 #5292 #FN = stein på båndet som ikke blir oppdaget.

prediction_cost = fp_cost + fn_cost #5537

# # y_train.shape (132,)
# y_val_pred_class.shape (132, 1)

# y_train[5]
# y_val_pred_class[5][1]
# y_train_index[5][0]

#show picture from FN in train:
for i in range(len(y_train)):
    if y_train[i] == 1 and y_val_pred_class[i][0] == 0:
        print(f"Value pred: {pred_train[i]}")
        print(picture_names_with_path[y_train_index[i]])
        img = Image.open(picture_names_with_path[y_train_index[i]])
        # img.show()