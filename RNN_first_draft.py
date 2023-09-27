import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from diverse_programs.Marian.no_onions import no_onions
import os

pictures = []
picture_names = []
path1 = "Bilder/256x192/Green/"
for image_i1 in os.listdir(path1):
    im = Image.open(path1+image_i1)
    print(image_i1)
    picture_names.append(image_i1)
    pictures.append(np.array(im))

path2 = "Bilder/256x192/Normal/"
for image_i2 in os.listdir(path2):
    im = Image.open(path2+image_i2)
    print(image_i2)
    picture_names.append(image_i2)
    pictures.append(np.array(im))

len(pictures)
len(picture_names)


X = np.array(pictures)
X.shape #(14, 192, 256, 3)

# data prep
X = X/255
X.shape

y = no_onions(picture_names) #[0,1,0,1,1,0]
y = np.array(y)
len(y)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape #(54400, 28, 28, 1)
X_test.shape #(13600, 28, 28, 1)
y_train.shape #(54400, 10)
y_test.shape #(13600, 10)

#expand X_train, X_test

X.shape
# create a model and train it:
input_layer = tf.keras.layers.Input(shape=(192,256,3)) #(None, 192, 256, 3)
hidden_layer1 = tf.keras.layers.Conv2D(10, (3,3), activation='relu')(input_layer) #(None, 254, 190, 10)
flatten_layer = tf.keras.layers.Flatten()(hidden_layer1) #(None, 482600)
dense_layer = tf.keras.layers.Dense(1, activation='softmax')(flatten_layer) #(None, 1)

print(f"shape input_layer {input_layer.shape}")
print(f"shape lstm_layer {hidden_layer1.shape}")
print(f"shape dense_layer {dense_layer.shape}")

my_model = tf.keras.models.Model(input_layer, dense_layer)
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

my_model.summary()

checkpoint_filepath = f"./TEMP_MODEL_WEIGHTS/RNN_first_draft"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    #save_best_only=True,
    save_weights_only=True,
    mode='max'
)

history = my_model.fit(X, y,
    batch_size=5,
    epochs=2,
    callbacks=[model_checkpoint_callback]
    )

my_model.load_weights(checkpoint_filepath)

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

my_model.save