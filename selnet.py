# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.util import nest
import numpy
from dataprepare import get_train_and_test_filename, get_batch_withlabels, get_selnet_batch
import tools

def slicing_where(condition, full_input, true_branch, false_branch):
    """Split `full_input` between `true_branch` and `false_branch` on `condition`.
     Args:
      condition: A boolean Tensor with shape [B_1, ..., B_N].
      full_input: A Tensor or nested tuple of Tensors of any dtype, each with
        shape [B_1, ..., B_N, ...], to be split between `true_branch` and
        `false_branch` based on `condition`.
      true_branch: A function taking a single argument, that argument having the
        same structure and number of batch dimensions as `full_input`. Receives
        slices of `full_input` corresponding to the True entries of
        `condition`. Returns a Tensor or nested tuple of Tensors, each with batch
        dimensions matching its inputs.
      false_branch: Like `true_branch`, but receives inputs corresponding to the
        false elements of `condition`. Returns a Tensor or nested tuple of Tensors
        (with the same structure as the return value of `true_branch`), but with
        batch dimensions matching its inputs.
    Returns:
      Interleaved outputs from `true_branch` and `false_branch`, each Tensor
      having shape [B_1, ..., B_N, ...].
    """
    full_input_flat = nest.flatten(full_input)
    true_indices = tf.where(condition)
    false_indices = tf.where(tf.logical_not(condition))
    true_branch_inputs = nest.pack_sequence_as(
        structure=full_input,
        flat_sequence=[tf.gather_nd(params=input_tensor, indices=true_indices)
                       for input_tensor in full_input_flat])
    false_branch_inputs = nest.pack_sequence_as(
        structure=full_input,
        flat_sequence=[tf.gather_nd(params=input_tensor, indices=false_indices)
                       for input_tensor in full_input_flat])
    true_outputs = true_branch(true_branch_inputs)
    false_outputs = false_branch(false_branch_inputs)
    nest.assert_same_structure(true_outputs, false_outputs)
    def scatter_outputs(true_output, false_output):
        batch_shape = tf.shape(condition)
        scattered_shape = tf.concat(
            [batch_shape, tf.shape(true_output)[tf.rank(batch_shape):]],
            0)
        true_scatter = tf.scatter_nd(
            indices=tf.cast(true_indices, tf.int32),
            updates=true_output,
            shape=scattered_shape)
        false_scatter = tf.scatter_nd(
            indices=tf.cast(false_indices, tf.int32),
            updates=false_output,
            shape=scattered_shape)
        return true_scatter + false_scatter
    result = nest.pack_sequence_as(
        structure=true_outputs,
        flat_sequence=[
            scatter_outputs(true_single_output, false_single_output)
            for true_single_output, false_single_output
            in zip(nest.flatten(true_outputs), nest.flatten(false_outputs))])
    return result

def shuffle(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

class Sel3DCNNConfig(object):
    
    def __init__(self):
        self.learning_rate = 0.01
        self.keep_prob = 1.0
        self.batch_size = 32
        self.epoch = 80
        self.cubic_shape = [10, 20, 20]

        self.test_size = 0.1
        self.seed = 121

class Sel3DCNN(object):

    def __init__(self, config):
        self.config = config
        self.learning_rate = config.learning_rate
        self.keep_prob = config.keep_prob
        self.batch_size = config.batch_size
        self.epoch_time = config.epoch
        self.batch_time = 0
        self.cubic_shape = config.cubic_shape

        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

        self.input_x = None
        self.selnet = None
        self.selnet_prediction = None
        self.selnet_label = None
        self.selnet_loss = None
        self.selnet_accuracy = None
        
        self.clsnet = None
        self.clsnet_prediction = None
        self.clsnet_label = None
        self.clsnet_loss = None
        self.clsnet_accuracy = None


    
    def CNN(self, input, shape):
        w = tf.Variable(tf.random_normal(shape, stddev=0.01), dtype=tf.float32)
        b = tf.Variable(tf.constant(0.01, shape=[shape[-1]]), dtype=tf.float32)
        out = tf.nn.relu(
            tf.add(tf.nn.conv3d(input, w, strides=[1, 1, 1, 1, 1], padding='VALID'), b))
        out = tf.nn.dropout(out, self.keep_prob)
        return out
    
    def FC(self, input, shape):
        w = tf.Variable(tf.random_normal(shape, stddev=0.01))
        out = tf.nn.relu(tf.add(tf.matmul(input, w), tf.constant(0.01, shape=[shape[-1]])))
        out = tf.nn.dropout(out, self.keep_prob)
        return out

    def archi_1(self, input):
        # return out_fc2
        with tf.name_scope("Archi1"):
            out_conv1 = self.CNN(input, [3, 5, 5, 1, 64])
            hidden_conv1 = tf.nn.max_pool3d(out_conv1, strides=[1, 1, 1, 1, 1], ksize=[1, 1, 1, 1, 1], padding='SAME')
            out_conv2 = self.CNN(out_conv1, [3, 5, 5, 64, 128])
            out_conv3 = self.CNN(out_conv2, [3, 5, 5, 128, 256])
            out_conv4 = self.CNN(out_conv3, [1, 1, 1, 256, 256])
            
            out_conv4 = tf.reshape(out_conv4, [-1, 256 * 8 * 8 * 4])
            out_fc1 = self.FC(out_conv4, [256 * 8 * 8 * 4, 200])
            out_fc2 = self.FC(out_fc1, [200, 2])

            out_conv4_shape = tf.shape(out_conv4)
            tf.summary.scalar('out_conv4_shape', tf.shape(out_conv3)[0])
            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', tf.shape(out_fc1)[0])
            return out_fc2

    def archi_2(self, input):
        # return out_fc2
        with tf.name_scope("Archi2"):
            out_conv1 = self.CNN(input, [3, 5, 5, 1, 64])
            hidden_conv1 = tf.nn.max_pool3d(out_conv1, strides=[1, 1, 1, 1, 1], ksize=[1, 1, 1, 1, 1], padding='SAME')
            out_conv2 = self.CNN(out_conv1, [3, 5, 5, 64, 128])
            out_conv3 = self.CNN(out_conv2, [3, 5, 5, 128, 256])
            out_conv4 = self.CNN(out_conv3, [1, 1, 1, 256, 256])
            
            out_conv4 = tf.reshape(out_conv4, [-1, 256 * 8 * 8 * 4])
            out_fc1 = self.FC(out_conv4, [256 * 8 * 8 * 4, 200])
            out_fc2 = self.FC(out_fc1, [200, 2])

            out_conv4_shape = tf.shape(out_conv4)
            tf.summary.scalar('out_conv4_shape', tf.shape(out_conv3)[0])
            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', tf.shape(out_fc1)[0])
            return out_fc2

    def select(self, input):
        with tf.name_scope("Select"):
            out_conv1 = self.CNN(input, [3, 5, 5, 1, 16])
            hidden_conv1 = tf.nn.max_pool3d(out_conv1, strides=[1, 1, 1, 1, 1], ksize=[1, 1, 1, 1, 1], padding='SAME')
            out_conv2 = self.CNN(out_conv1, [3, 5, 5, 16, 32])
            out_conv3 = self.CNN(out_conv2, [3, 5, 5, 32, 64])
            out_conv4 = self.CNN(out_conv3, [1, 1, 1, 64, 64])
            out_conv4 = tf.reshape(out_conv4, [-1, 64 * 8 * 8 * 4])
            
            out_fc1 = self.FC(out_conv4, [64 * 8 * 8 * 4, 200])
            out_fc2 = self.FC(out_fc1, [200, 2])

            out_conv4_shape = tf.shape(out_conv4)
            tf.summary.scalar('out_conv4_shape', tf.shape(out_conv3)[0])
            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', tf.shape(out_fc1)[0])
            return out_fc2
    
    def setup_graph(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.cubic_shape[0], self.cubic_shape[1], self.cubic_shape[2]])
        input = tf.reshape(self.input_x, [-1, self.cubic_shape[0], self.cubic_shape[1], self.cubic_shape[2], 1])
        selnet = self.select(input)
        self.selnet = selnet
        self.selnet_prediction = tf.argmax(tf.nn.sigmoid(selnet), 1)
        self.selnet_label = tf.placeholder(tf.float32, [None, 2])
        self.selnet_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=selnet, labels=self.selnet_label))
        self.selnet_accuracy = tf.reduce_mean(tf.cast(tf.equal(\
            self.selnet_prediction, tf.argmax(self.selnet_label, 1)), tf.float32))

        clsnet = slicing_where(
            condition=tf.equal(self.selnet_prediction, 0),
            full_input=input,
            true_branch=lambda x: self.archi_1(x),
            false_branch=lambda x: self.archi_2(x))  
        self.clsnet = clsnet
        self.clsnet_prediction = tf.argmax(tf.nn.sigmoid(clsnet), 1)
        self.clsnet_label = tf.placeholder(tf.float32, [None, 2])
        self.clsnet_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=clsnet, labels=self.clsnet_label))
        self.clsnet_accuracy = tf.reduce_mean(tf.cast(tf.equal(\
            self.clsnet_prediction, tf.argmax(self.clsnet_label, 1)), tf.float32))

        return selnet, clsnet
    
    def prepare_selnet_data(self, base_dir):
        dir = tools.workspace(base_dir)
        train_filenames, test_filenames = get_train_and_test_filename(dir.npy_path, self.config.test_size, self.config.seed)
        self.train_data, self.train_label, _ = get_selnet_batch(dir.npy_path, train_filenames)
        self.test_data, self.test_label, _ = get_selnet_batch(dir.npy_path, test_filenames)
        self.batch_time = len(train_filenames) // self.batch_size
        if (len(train_filenames) % self.batch_size) != 0:
            self.batch_time += 1
    
    def prepare_clsnet_data(self, base_dir):
        dir = tools.workspace(base_dir)
        train_filenames, test_filenames = get_train_and_test_filename(dir.npy_path, self.config.test_size, self.config.seed)
        self.train_data, self.train_label, _ = get_batch_withlabels(dir.npy_path, train_filenames)
        self.test_data, self.test_label, _ = get_batch_withlabels(dir.npy_path, test_filenames)
        self.batch_time = len(train_filenames) // self.batch_size
        if (len(train_filenames) % self.batch_size) != 0:
            self.batch_time += 1

    def next_epoch(self):
        self.train_data, self.train_label = shuffle(self.train_data, self.train_label)
        self.batch_index = 0
    
    def next_batch(self):
        i = self.batch_index
        self.batch_index += 1
        batch_data = self.train_data[i*self.batch_size: (i+1)*self.batch_size]
        batch_label = self.train_label[i*self.batch_size: (i+1)*self.batch_size]
        return batch_data, batch_label
    
    def test_batch(self):
        return self.test_data, self.test_label 

if __name__ == '__main__':
    pass
