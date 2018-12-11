import json
import numpy as np
import tensorflow
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model, Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
IMG_SIZE = (256, 256)
core_idg = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip=True,
                              vertical_flip=False,
                              height_shift_range=0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range=0.1,
                              fill_mode='reflect',
                              zoom_range=0.15)


def flow_from_dataframe(img_data_gen, image_path, **dflow_args):
    print('## Applying Image Transformation')
    df_gen = img_data_gen.apply_transform(image_path,
                                          **dflow_args)
    # df_gen.filenames = in_df[path_col].values
    # df_gen.classes = np.stack(in_df[y_col].values)
    # df_gen.samples = in_df.shape[0]
    # df_gen.n = in_df.shape[0]
    # df_gen._set_index_array()
    # df_gen.directory = ''  # since we have the full path
    # print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


def create_model(all_labels):
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(64, (3, 3), input_shape=(256, 256, 1), activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(Flatten())
    # Step 3 - Dense layer
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=len(all_labels), activation='sigmoid'))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()
    return classifier


with open('all_labels.json') as f:
    all_labels = json.load(f)
print(len(all_labels))

classifier = create_model(all_labels)
classifier.load_weights('model_2.h5')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def predict(test):
    img = image.load_img(test, target_size=(256, 256))
    img = image.img_to_array(img)
    img = rgb2gray(img)
    img = np.expand_dims(img, axis=0)
    img = np.reshape(img, (256, 256, 1))
    img = [
        [img]]
    pred_y = classifier.predict(img)
    get_index = [i for i, x in enumerate(pred_y[0]) if x]
    disease_labels_predicted = [all_labels[i] for i in get_index]
    if len(disease_labels_predicted) == 0:
        import random
        disease_labels_predicted = [random.choice(all_labels)]
    return disease_labels_predicted
