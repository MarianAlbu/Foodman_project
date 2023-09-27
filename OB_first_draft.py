import itertools
import os
from PIL import Image

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

model_name = "efficientnetv2-xl-21k"
IMAGE_SIZE = (192, 256)

data_dir = 'C:\\Users\\Jeanette\\OneDrive\\Innleveringer\\00 Graduate project\\Project on Git\\GraduationProject\\Bilder\\256x192'
#data_dir = 'C:\\Users\\Jeanette\\.keras\\datasets\\flower_photos'

def build_dataset(subset):
  return tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=.20,
      subset=subset,
      label_mode="categorical",
      # Seed needs to provided when using validation_split and shuffle = True.
      # A fixed seed is used so that the validation set is stable across runs.
      seed=123,
      image_size=IMAGE_SIZE,
      batch_size=1)

train_ds = build_dataset("training")
class_names = tuple(train_ds.class_names)
train_size = train_ds.cardinality().numpy()
train_ds = train_ds.unbatch().batch(BATCH_SIZE)
train_ds = train_ds.repeat()


pictures = []
picture_names = []
path1 = "Bilder/256x192/Green/"
for image_i1 in os.listdir(path1):
    im = Image.open(path1+image_i1)
    print(image_i1)
    picture_names.append(image_i1)
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