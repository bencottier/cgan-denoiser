# -*- coding: utf-8 -*-
"""
test_undersampled.py

Train a neural network to remove artefacts caused by undersampling.
We define undersampling as sampling at a frequency below the Nyquist limit.
Methods include compressed sensing and chaotic sensing.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
import artefacts
import data_processing
from config import Config as config
import cgan as model
import utils
import tensorflow as tf
tf.enable_eager_execution()
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import math


def generator_d_loss(generated_output):
    # [1,1,...,1] with generated images since we want the discriminator to judge them as real
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def generator_abs_loss(labels, generated_images):
    # As well as "fooling" the discriminator, we want particular pressure on ground-truth accuracy
    return config.L1_lambda * tf.losses.absolute_difference(labels, generated_images)  # mean


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def train_step(inputs, labels):
    with tf.GradientTape() as gen_tape:
        generated_images = generator(inputs, training=True)

        gen_loss = generator_abs_loss(labels, generated_images)
        gen_rmse = data_processing.rmse(labels, generated_images)
        gen_psnr = data_processing.psnr(labels, generated_images)

        # Logging
        global_step.assign_add(1)
        log_metric(gen_loss, "train/loss/generator")
        log_metric(gen_rmse, "train/accuracy/rmse")
        log_metric(gen_psnr, "train/accuracy/psnr")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))


def validate_step(inputs, labels):
    generated_images = generator(inputs, training=False)

    gen_loss = generator_abs_loss(labels, generated_images)
    gen_rmse = data_processing.rmse(labels, generated_images)
    gen_psnr = data_processing.psnr(labels, generated_images)

    # Logging
    global_step.assign_add(1)
    log_metric(gen_loss, "valid/loss/generator")
    log_metric(gen_rmse, "valid/accuracy/rmse")
    log_metric(gen_psnr, "valid/accuracy/psnr")


def train(train_dataset, valid_dataset=None, epochs=1):
    for epoch in range(epochs):
        start = time.time()

        for x, y in train_dataset.make_one_shot_iterator():
            train_step(x, y)

        for x, y in valid_dataset.make_one_shot_iterator():
            validate_step(x, y)

        generate_and_save_images(generator,
                                 epoch + 1,
                                 selected_inputs,
                                 selected_labels)
        
        # saving (checkpoint) the model every few epochs
        if (epoch + 1) % config.save_per_epoch == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                          time.time()-start))
    # generating after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             selected_inputs,
                             selected_labels)


def test(model, inputs, labels):
    print("Running model on test set...")
    prediction_time = np.zeros(inputs.shape[0], dtype=float)
    rmse = np.zeros(inputs.shape[0], dtype=np.float64)
    psnr = np.zeros(inputs.shape[0], dtype=np.float64)
    for i in range(inputs.shape[0]):
        t0 = time.time()
        prediction = generator(inputs[i:i+1], training=False)
        prediction_time[i] = time.time() - t0
        rmse[i] = data_processing.rmse(labels[i:i+1], prediction, norm=2)
        psnr[i] = data_processing.psnr(labels[i:i+1], prediction, max_diff=1)
    print("Results over test set:")
    print("Time mean: {:.4f} sec".format(np.mean(prediction_time)))
    print("Time stdv: {:.4f} sec".format(math.sqrt(np.var(prediction_time))))
    print("RMSE mean: {:.4f}".format(np.mean(rmse)))
    print("RMSE stdv: {:.4f}".format(math.sqrt(np.var(rmse))))
    print("PSNR mean: {:.4f} dB".format(np.mean(psnr)))
    print("PSNR stdv: {:.4f} dB".format(math.sqrt(np.var(psnr))))


def generate_and_save_images(model, epoch, test_inputs, test_labels):
    if model is None:
        predictions = test_inputs
    else:
        # Make sure the training parameter is set to False because we
        # don't want to train the batchnorm layer when doing inference.
        predictions = model(test_inputs, training=False)

    types = [predictions, test_labels]  # Image types (alternated in rows)
    ntype = len(types)
    nrows = 4
    ncols = 8
    fig = plt.figure(figsize=(8, 5))
    
    for i in range(ntype * predictions.shape[0]):
        plt.subplot(nrows, ncols, i+1)
        # Get relative index
        row = int(i / ncols)
        row_rel = row % ntype
        group = int(row / ntype)
        shift = ncols * (group * (ntype - 1) + row_rel)
        idx = i - shift
        # Plot
        for t in range(ntype):
            if row_rel == 0:
                j = int(i / ntype)
                rmse = data_processing.rmse(test_labels[j], predictions[j], norm=2)
                psnr = data_processing.psnr(test_labels[j], predictions[j], max_diff=1)
                plt.xlabel('RMSE={:.3f}\nPSNR={:.2f}'.format(rmse, psnr), fontsize=8)
            if row_rel == t:
                plt.imshow(types[row_rel][idx, :, :, 0], vmin=-1, vmax=1, cmap='gray')
                break
        plt.xticks([])
        plt.yticks([])
    
    plt.savefig(os.path.join(results_path, 'image_at_epoch_{:04d}.png'.format(epoch)))  # TODO
    # plt.show()  # TODO


def plot_samples(model, inputs, labels, n_samples):
    """
    Plot samples of a dataset with comparison and error.
    """
    print("Plotting samples")
    prediction = model(inputs, training=False)
    for i in range(n_samples):
        fig = plt.figure(figsize=(9, 7))
        plt.subplot(2, 2, 1)
        plt.imshow(inputs[i, :, :, 0], vmin=-1, vmax=1, cmap='gray')
        plt.title('Zero-fill')
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(prediction[i, :, :, 0], vmin=-1, vmax=1, cmap='gray')
        plt.title('Conditional GAN')
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(labels[i, :, :, 0], vmin=-1, vmax=1, cmap='gray')
        plt.title('Ground truth')
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.imshow(labels[i, :, :, 0], vmin=-1, vmax=1, cmap='gray')
        plt.imshow(np.abs(labels[i, :, :, 0] - prediction[i, :, :, 0]), cmap='jet', alpha=0.3)
        plt.title('Abs. error overlay')
        plt.colorbar()
        plt.show()


def log_metric(value, name):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar(name, value)


if __name__ == '__main__':
    # model_path = "out/fractal_oasis1_cgan/model/2018-12-17-16-20-36"
    # results_path = "out/fractal_oasis1_cgan/results/2018-12-17-16-20-36"

    # Make directories for this run
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_path = os.path.join(config.model_path, time_string)
    results_path = os.path.join(config.results_path, time_string)
    utils.safe_makedirs(model_path)
    utils.safe_makedirs(results_path)

    # Initialise logging
    log_path = os.path.join('logs', config.exp_name, time_string)
    summary_writer = tf.contrib.summary.create_file_writer(log_path, flush_millis=10000)
    summary_writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()

    # Load the data
    # Train and test
    (train_inputs, train_labels), (test_inputs, test_labels), case_list = data_processing.get_oasis_dataset(
        config.input_path, config.label_path, 
        config.test_cases, config.max_training_cases, config.train_size)
    # Test only
    # test_inputs, test_labels = data_processing.get_oasis_dataset_test(
    #     config.input_path, config.label_path, 
    #     config.test_cases, len(config.test_cases), config.train_size)

    # Training set
    full_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    full_dataset.shuffle(config.buffer_size)
    valid_size = int(config.validation_split * train_inputs.shape[0])
    train_dataset = full_dataset.skip(valid_size).batch(config.batch_size)
    valid_dataset = full_dataset.take(valid_size).batch(config.batch_size)

    # Test set
    # Set up some random (but consistent) test cases to monitor
    n_test_samples = 16
    random_indices = np.random.choice(np.arange(test_inputs.shape[0]),
                                      n_test_samples,
                                      replace=False)
    selected_inputs = test_inputs[random_indices]
    selected_labels = test_labels[random_indices]
    
    # Set up the models for training
    generator = model.make_generator_model()

    generator_optimizer = tf.train.AdamOptimizer(config.learning_rate)

    checkpoint_prefix = os.path.join(model_path, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=generator_optimizer, model=generator)

    # Train
    generate_and_save_images(None, 0, selected_inputs, selected_labels)  # baseline
    print("\nTraining...\n")
    # Compile training function into a callable TensorFlow graph (speeds up execution)
    train_step = tf.contrib.eager.defun(train_step)
    train(train_dataset, valid_dataset, config.max_epoch)
    print("\nTraining done\n")

    # Test
    checkpoint.restore(tf.train.latest_checkpoint(model_path))
    test(generator, test_inputs, test_labels)
    # plot_samples(generator, selected_inputs, selected_labels, n_test_samples)
    # generate_and_save_images(generator, config.max_epoch + 1, selected_inputs, selected_labels)

    print("End of main program")
