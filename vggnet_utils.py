import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

import pandas as pd
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import fbeta_score

VGG_MEAN = [123.68, 116.78, 103.94]

"""
Setup:

Uses tf.contrib.data module which is in release candidate 1.2.0rc0
Based on:
    - PyTorch example from Justin Johnson:
      https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
      - https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c
Required packages: tensorflow (v1.2)
You can install the release candidate 1.2.0rc0 here:
https://www.tensorflow.org/versions/r1.2/install/

Download the weights trained on ImageNet for VGG:
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```
"""

def testing():
    print("done")
    return


def list_images(directory):
    """ Get all the images and labels from data/train_v2.csv"""
    
    filenames_targets = pd.read_csv('data/train_v2.csv') # column1: image_name, column2: tags (labels for image file)
    filenames = [directory + file + '.jpg' for file in filenames_targets['image_name'].tolist()]
    
    labels = filenames_targets['tags'].tolist()
    
    # Convert to one-hot labels
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in filenames_targets['tags'].values])))
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    one_hot_labels = []
    
    for f, tags in tqdm(filenames_targets.values, miniters=1000):
        targets = [0]*17 #np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1 
        one_hot_labels.append(targets)
    print("listed")
    return filenames, one_hot_labels # [:1000]
    

# change to F score
'''
def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc
'''


def split_samples(all_filenames, all_labels):
    """ Split all filenames and labels into training and test sets, return both"""
    
    n = len(all_filenames)
    order = random.sample(range(n), n)
    all_filenames_random = [all_filenames[i] for i in order]
    all_labels_random = [all_labels[i] for i in order]
    
    third = int(n/3)
    val_filenames = all_filenames_random[:third]
    val_labels = all_labels_random[:third]
    train_filenames = all_filenames_random[third:]
    train_labels = all_labels_random[third:]
    
    return train_filenames, train_labels, val_filenames, val_labels


# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                     lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label


# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def training_preprocess(image, label):
    crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)
    return centered_image, label


# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)

    return centered_image, label

    
def define_graph(args):
    """Defines the computational graph for the transfer learning model"""
        
    # Get the list of filenames and corresponding list of labels for training et validation
    # train_filenames, train_labels = list_images(args.train_dir)
    # val_filenames, val_labels = list_images(args.val_dir)

    all_filenames, all_labels = list_images(args.train_dir)
    train_filenames, train_labels, val_filenames, val_labels = split_samples(all_filenames, all_labels)
    num_classes = 17

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
    # Standard preprocessing for VGG on ImageNet taken from here:
    # https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
    # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

    # ----------------------------------------------------------------------
    # DATASET CREATION using tf.contrib.data.Dataset
    # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

    # The tf.contrib.data.Dataset framework uses queues in the background to feed in
    # data to the model.
    # We initialize the dataset with a list of filenames and labels, and then apply
    # the preprocessing functions described above.
    # Behind the scenes, queues will load the filenames, preprocess them with multiple
    # threads and apply the preprocessing in parallel, and then batch the data

        # Training dataset
        train_filenames = tf.constant(train_filenames)
        train_labels = tf.constant(train_labels)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(_parse_function,
           num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.map(training_preprocess,
           num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(args.batch_size)
    
        # Validation dataset
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_parse_function,
        num_threads=args.num_workers, output_buffer_size=args.batch_size)
        val_dataset = val_dataset.map(val_preprocess,
        num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_val_dataset = val_dataset.batch(args.batch_size)
    
        print("dataset created")
        
        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.
    
        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()
        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)
    
        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)
    
        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                                       dropout_keep_prob=args.dropout_keep_prob)
    
        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))
    
        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
    
        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        fc8_init = tf.variables_initializer(fc8_variables)
    
        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        # tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) 
        tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits) # softmax cross entropy loss so can have labels with multiple classes
        loss = tf.losses.get_total_loss()  
    
        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
        fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)
    
        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
        full_train_op = full_optimizer.minimize(loss)
        
        tf.get_default_graph().finalize()
    
    return graph, init_fn, fc8_init, fc8_optimizer, fc8_train_op, loss

    
    
    
    