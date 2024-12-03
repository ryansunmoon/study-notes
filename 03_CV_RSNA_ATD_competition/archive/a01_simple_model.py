import pandas as pd
# import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = '/Users/ryansun/Desktop/DS/data/rsna_atd'
pic_path = pic_paths = glob.glob(os.path.join(data, 'train_images', '*', '*', '*.png'))
# len(pic_path)
# 11,536

df_train_label = pd.read_csv(os.path.join(data, 'train.csv'))

df_train_label['image_path_2']= df_train_label['image_path'].str.replace(
    "/kaggle/input/rsna-2023-abdominal-trauma-detection/", '/Users/ryansun/Desktop/DS/data/rsna_atd/')

df_train_label['image_path_2']= df_train_label['image_path_2'].str.replace(
    ".dcm", '.png')

df_train = df_train_label[['bowel_injury', 'image_path_2']].drop_duplicates()

df_train.head()
df_train['bowel_injury']= df_train['bowel_injury'].astype(str)


df_train['image_path_2'][0]

# build the simple model;
# this gets the data in. great
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = df_train,
    directory=None,
    x_col='image_path_2',
    y_col='bowel_injury',
    weight_col=None,
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=1,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    subset='training',
    interpolation='nearest',
    validate_filenames=True)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe = df_train,
    directory=None,
    x_col='image_path_2',
    y_col='bowel_injury',
    weight_col=None,
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=1,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    subset='validation',
    interpolation='nearest',
    validate_filenames=True)

image = validation_generator.next()

plt.imshow(image[0][0])
plt.show()

import tensorflow as tf
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.optimizers.legacy import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])
#
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator)


# model.evaluate(test_images, test_labels)
# classifications = model.predict(test_images)
# print(classifications[0])
# print(test_labels[0])

# the following is to generate the graph

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
