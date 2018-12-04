# -*- coding: utf-8 -*-
"""
UNet 2D Keras Cartilage Segmentation.

Initial UNet and DSC functions from github.com/jocicmarko/ultrasound-nerve-segmentation. MIT License

Created on Wed May 23 13:56:26 2018

@author: uqscha22
"""
from __future__ import print_function, division
import data_processing as dp
import filenames
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import nibabel as nib #loading Nifti images
import time


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_numpy(y_true, y_pred):
    smooth = 1.
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def rms(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def rms_numpy(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def rms_loss(y_true, y_pred):
    return rms(y_true, y_pred)


def get_unet(img_rows, img_cols):
    '''
    Returns UNet using Keras
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='linear')(conv9)  # was sigmoid

    model = Model(inputs=[inputs], outputs=[conv10])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss=rms_loss, metrics=[rms])

    return model


def get_unet_lowres(img_rows, img_cols):
    '''
    Returns UNet (with lower resolution than normal) using Keras
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='linear')(conv9)  # was sigmoid

    model = Model(inputs=[inputs], outputs=[conv10])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss=rms_loss, metrics=[rms])

    return model

if __name__ == '__main__':
    
    # Parameters
    # test_cases = [1, 3, 4, 6, 10, 12, 50, 100, 120, 140, 145, 150, 
    #     151, 152, 160, 200, 210, 300, 350, 351]
    test_cases = [1, 3, 4, 6, 10, 12, 50, 100, 120, 140, 145, 150, 151, 152]
    batch_size = 10
    max_training_cases = 150
    validation_split = 0.2
    size = 256
    iterations = 100
    
    # Load MR data
    root_path = "/home/ben/projects/honours/datasets/oasis1/"
    input_path = root_path + "slices_artefact/"  # "slices_pad/"
    label_path = root_path + "slices_pad/"  # "slices_seg_pad/"

    (inputs, labels), (test_inputs, test_labels), case_list = dp.get_dataset(
        input_path, label_path, test_cases, max_training_cases, size)
    rows, cols = inputs.shape[1:3]

    # model = get_unet(rows, cols)
    model = get_unet_lowres(rows, cols)
    '''
    print("Training ...")
    model_checkpoint = ModelCheckpoint('model/weights.h5', 
        monitor='val_loss', save_best_only=True)
    start = time.time() #time generation
    model.fit(inputs, labels, batch_size=batch_size, epochs=iterations, 
              verbose=1, shuffle=True,
              validation_split=validation_split,
              callbacks=[model_checkpoint])  # with callback
    # model.fit(imgs, labels, batch_size=batch_size, epochs=iterations, 
    #           verbose=1, shuffle=True,
    #           validation_split=validation_split)  # without callback
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    '''
    print('Loading best saved weights...')  # last may not be best one
    model.load_weights('model/weights.h5')
    
    print("Testing ...")
    start = time.time()  # time generation
    prediction = model.predict(test_inputs, verbose=1)
    end = time.time()
    elapsed = end - start
    print("Prediction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    metrics = []
    for index, val in enumerate(test_cases):
        # metric = dice_coef_numpy(testLabels[index].astype(np.float32), prediction[index])
        metric = rms_numpy(test_labels[index].astype(np.float32), prediction[index])
        metrics.append(metric)
        print("Metric", case_list[val], metric)
    metrics = np.array(metrics)
    print("Mean metric:", np.mean(metrics))
    print("Stddev metric:", np.std(metrics))
    print("Median metric:", np.median(metrics))
    
    # Plot results
    import matplotlib.pyplot as plt

    for index, val in enumerate(test_cases):
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.gray()
        plt.tight_layout()
        plt.subplot(131)
        plt.title('Image ' + str(case_list[val]))
        plt.imshow(dp.normalise(test_inputs[index,:,:,0]*127.5 - 127.5))
        plt.subplot(132)
        plt.title('Label ' + str(case_list[val]))
        plt.imshow(dp.normalise(test_labels[index,:,:,0]))
        plt.subplot(133)
        plt.title('Prediction ' + str(case_list[val]))
        plt.imshow(dp.normalise(prediction[index,:,:,0]))
        plt.show()
    