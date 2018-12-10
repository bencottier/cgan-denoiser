# -*- coding: utf-8 -*-
"""
test_cgan.py

Testing a Conditional GAN model.

author: Benjamin Cottier
"""
from __future__ import absolute_import, division, print_function
import artefacts
import data_processing
from config import ConfigCGAN as config
import cgan as model
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


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


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
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(inputs, training=True)

        real_output = discriminator(labels, training=True)
        generated_output = discriminator(generated_images, training=True)
            
        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

        # Logging
        global_step.assign_add(1)
        log_loss(gen_loss, "generator")
        log_loss(disc_loss, "discriminator")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for x, y in dataset.make_one_shot_iterator():
            train_step(x, y)

        generate_and_save_images(generator,
                                 epoch + 1,
                                 selected_inputs,
                                 selected_labels)
        
        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % config.save_per_epoch == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                          time.time()-start))
    # generating after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             selected_inputs,
                             selected_labels)


def generate_and_save_images(model, epoch, test_inputs, test_labels):
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
            if row_rel == ntype -  1:
                j = int(i / ntype)
                rmse = data_processing.rmse(test_labels[j], predictions[j], norm=2)
                psnr = data_processing.psnr(test_labels[j], predictions[j], max_diff=1)
                plt.xlabel('RMSE={:.3f}\nPSNR={:.2f}'.format(rmse, psnr), fontsize=8)
            if row_rel == t:
                plt.imshow(types[row_rel][idx, :, :, 0] * 127.5 + 127.5, cmap='gray')
                break
        plt.xticks([])
        plt.yticks([])
    
    plt.savefig(os.path.join(config.data_path, 
                             'image_at_epoch_{:04d}.png'.format(epoch)))
    # plt.show()


def log_loss(loss, name):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("loss_" + name, loss)


if __name__ == '__main__':
    # Initialise logging
    log_dir = os.path.join('logs', config.exp_name, time.strftime("%Y-%m-%d-%H-%M-%S"))
    summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
    summary_writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()

    # Load the dataset
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 
                                        config.raw_size,
                                        config.raw_size,
                                        config.channels)
    # Add noise for condition input
    train_images = data_processing.normalise(train_images, (-1, 1), (0, 255))
    train_inputs = artefacts.add_gaussian_noise(train_images, stdev=0.2).astype('float32')
    train_labels = train_images.astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))\
        .shuffle(config.buffer_size).batch(config.batch_size)
    
    # Set up the models for training
    generator = model.make_generator_model()
    discriminator = model.make_discriminator_model()

    generator_optimizer = tf.train.AdamOptimizer(config.learning_rate)
    discriminator_optimizer = tf.train.AdamOptimizer(config.learning_rate)

    checkpoint_prefix = os.path.join(config.model_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    # Set up some random (but consistent) cases to monitor over training
    num_examples_to_generate = 16
    random_indices = np.random.choice(np.arange(train_inputs.shape[0]),
                                      num_examples_to_generate,
                                      replace=False)
    selected_inputs = train_inputs[random_indices]
    selected_labels = train_labels[random_indices]

    print("\nTraining...\n")
    # Compile training function into a callable TensorFlow graph (speeds up execution)
    train_step = tf.contrib.eager.defun(train_step)
    train(train_dataset, config.max_epoch)
    print("\nTraining done\n")

    # checkpoint.restore(tf.train.latest_checkpoint(config.model_path))
    # generate_and_save_images(generator, 0, selected_inputs, selected_labels)
