# -*- coding:utf-8 -*-
"""
 the idea of this script came from LUNA2016 champion paper.
 This model conmposed of three network,namely Archi-1(size of 10x10x6),Archi-2(size of 30x30x10),Archi-3(size of 40x40x26)

input: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half. Shape [batch, in_depth, in_height, in_width, in_channels].
filter: A Tensor. Must have the same type as input. Shape [filter_depth, filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
strides: A list of ints that has length >= 5. 1-D tensor of length 5. The stride of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1.
padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
name: A name for the operation (optional).

"""
import tensorflow as tf
from dataprepare import get_batch, get_train_and_test_filename, get_batch_withlabels, get_high_data, \
    get_batch_withlabels_high
import random
import time
import datetime
import os
import tools
from tools import log, logtime, count_high, count_low

class model(object):

    def __init__(self, learning_rate, keep_prob, batch_size, epoch):
        print(" network begin...")
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.epoch = epoch

        self.cubic_shape = [[10, 20, 20], [6, 20, 20], [26, 40, 40]]

    def archi_1(self, input, sphericity, margin, lobulation, spiculation, keep_prob):
        # return out_fc2
        with tf.name_scope("Archi-1"):
            # input size is batch_sizex20x20x6
            # 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3, 5, 5, 1, 64], stddev=0.01), dtype=tf.float32, name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv1')
            out_conv1 = tf.nn.relu(
                tf.add(tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1, keep_prob)

            # max pooling ,pooling layer has no effect on the data size
            hidden_conv1 = tf.nn.max_pool3d(out_conv1, strides=[1, 1, 1, 1, 1], ksize=[1, 1, 1, 1, 1], padding='SAME')

            # after conv1 ,the output size is batch_sizex4x16x16x64([batch_size,in_deep,width,height,output_deep])
            w_conv2 = tf.Variable(tf.random_normal([3, 5, 5, 64, 128], stddev=0.01), dtype=tf.float32, name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[128]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(
                tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)

            # after conv2 ,the output size is batch_sizex2x12x12x64([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([3, 5, 5, 128, 256], stddev=0.01), dtype=tf.float32, name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[256]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(
                tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)

            w_conv4 = tf.Variable(tf.random_normal([1, 1, 1, 256, 256], stddev=0.01), dtype=tf.float32, name='w_conv4')
            b_conv4 = tf.Variable(tf.constant(0.01, shape=[256]), dtype=tf.float32, name='b_conv4')
            out_conv4 = tf.nn.relu(
                tf.add(tf.nn.conv3d(out_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv4))
            out_conv4 = tf.nn.dropout(out_conv4, keep_prob)

            out_conv3_shape = tf.shape(out_conv4)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

            # after conv2 ,the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv4 = tf.reshape(out_conv4, [-1, 256 * 8 * 8 * 4])
            w_fc1 = tf.Variable(tf.random_normal([256 * 8 * 8 * 4, 200], stddev=0.01), name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv4, w_fc1), tf.constant(0.01, shape=[200])))
            out_fc1 = tf.nn.dropout(out_fc1, keep_prob)

            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([200, 2], stddev=0.01), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.01, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)
            # sphericity, margin, lobulation, spiculation
            # w_sphericity = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_sphericity')
            # w_margin = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_margin')
            # w_lobulation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_lobulation')
            # w_spiculation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_spiculation')

            # b_sphericity = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_sphericity')
            # b_margin = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_margin')
            # b_lobulation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_lobulation')
            # b_spiculation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_spiculation')

            # out_sphericity = tf.nn.relu(tf.add(tf.matmul(sphericity, w_sphericity), b_sphericity))
            # out_margin = tf.nn.relu(tf.add(tf.matmul(margin, w_margin), b_margin))
            # out_lobulation = tf.nn.relu(tf.add(tf.matmul(lobulation, w_lobulation), b_lobulation))
            # out_spiculation = tf.nn.relu(tf.add(tf.matmul(spiculation, w_spiculation), b_spiculation))

            # out_fc2 = tf.add(out_fc2, out_sphericity)
            # out_fc2 = tf.add(out_fc2, out_margin)
            # out_fc2 = tf.add(out_fc2, out_lobulation)
            # out_fc2 = tf.add(out_fc2, out_spiculation)
            # print(out_fc2)
            return out_fc2

    def inference(self, base_dir, model_index, test_size, seed, train_flag=True):
        # some statistic index
        highest_acc = 0.0
        highest_iterator = 1

        dir = tools.workspace(base_dir, self.keep_prob)

        train_filenames, test_filenames = get_train_and_test_filename(dir.npy_path, test_size, seed)
        # how many time should one epoch should loop to feed all data
        times = len(train_filenames) // self.batch_size
        if (len(train_filenames) % self.batch_size) != 0:
            times = times + 1

        tf.logging.set_verbosity(tf.logging.ERROR)
        log('input npy dir: %s' % dir.npy_path)
        log('cubic shape: (%d, %d, %d)' % (self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2]))
        log("Training examples: %d, high %d, low %d" % (len(train_filenames), count_high(train_filenames), count_low(train_filenames)))
        log("Testing examples: %d, high %d, low %d" % (len(test_filenames), count_high(test_filenames), count_low(test_filenames)))
        log('epoch: %d' % self.epoch)
        log('batch size: %d' % self.batch_size)
        log('keep prob: %f' % self.keep_prob)
        logtime('start time: ')
        # keep_prob used for dropout
        keep_prob = tf.placeholder(tf.float32)
        # take placeholder as input
        x = tf.placeholder(tf.float32, [None, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1],
                                        self.cubic_shape[model_index][2]])

        # <sphericity>3</sphericity>
        # <margin>3</margin>
        # <lobulation>3</lobulation>
        # <spiculation>4</spiculation>
        sphericity = tf.placeholder(tf.float32)
        margin = tf.placeholder(tf.float32)
        lobulation = tf.placeholder(tf.float32)
        spiculation = tf.placeholder(tf.float32)

        # X = tf.placeholder(tf.float32)
        # Y = tf.placeholder(tf.float32)

        # W = tf.Variable(tf.random_normal([1]), name='weight')
        # b = tf.Variable(tf.random_normal([1]), name='bias')

        x_image = tf.reshape(x, [-1, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1],
                                 self.cubic_shape[model_index][2], 1])
        net_out = self.archi_1(x_image, sphericity, margin, lobulation, spiculation, keep_prob)
        # print(net_out)
        # save all epoch results
        saver = tf.train.Saver(max_to_keep=self.epoch)  # default to save all variable,save mode or restore from path

        if train_flag:
            global_step = tf.Variable(0)

            learning_rate = tf.train.exponential_decay(0.01, global_step, times * 40, 1, staircase=True)

            # softmax layer
            real_label = tf.placeholder(tf.float32, [None, 2])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label)
            # cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
            net_loss = tf.reduce_mean(cross_entropy)

            train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(net_loss)

            prediction = tf.nn.sigmoid(net_out)
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
            accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            merged = tf.summary.merge_all()

            # allow memory allocation growth
            # config = tf.ConfigProto()
            # config.gpu_options.per_process_gpu_memory_fraction = 0.4
            # config.gpu_options.allow_growth = True
            with tf.Session() as sess:
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                sess.run(tf.global_variables_initializer())
                train_writer = tf.summary.FileWriter(dir.tensorboard_path, sess.graph)
                # loop epoches
                for i in range(self.epoch):
                    epoch_start = time.time()
                    #  the data will be shuffled by every epoch
                    random.shuffle(train_filenames)
                    # times = 5
                    for t in range(times):
                        print('\r', ('%d/%d' % (t+1, times)).ljust(10), end='', flush=True)

                        batch_files = train_filenames[t * self.batch_size:(t + 1) * self.batch_size]
                        batch_data, sphericityt, margint, lobulationt, spiculationt, batch_label = \
                            get_batch_withlabels(dir.npy_path, batch_files)
                        feed_dict = {x: batch_data, sphericity: sphericityt, margin: margint, lobulation: lobulationt,
                                     spiculation: spiculationt, real_label: batch_label, keep_prob: self.keep_prob}
                        _, summary, out_res = sess.run([train_step, merged, net_out], feed_dict=feed_dict)
                        # print(len(out_res))
                        # print(out_res[0])
                        # print(sess.run(tf.nn.sigmoid(out_res[0])))
                        # feed_dict = {global_step: t + i * times}
                        lnrt = sess.run(learning_rate, feed_dict={global_step: t + i * times})
                        if t == times - 1:
                            print('\r---------------------------------------------------')
                            print('learning rate: ', lnrt)
                        train_writer.add_summary(summary, i)

                    epoch_end = time.time()
                    # randomtestfiles = random.sample(test_filenames, 32)
                    test_batch, sphericityt, margint, lobulationt, spiculationt, test_label = \
                        get_batch_withlabels(dir.npy_path, test_filenames)
                    # print(test_label)

                    x10 = 0
                    x01 = 0
                    for label in test_label:
                        if label[0] == 1:
                            x10 += 1
                        else:
                            x01 += 1
                    print('percent: ', x10 / len(test_label))
                    # f.write('percent: %f ' % x10 / 16)
                    test_dict = {x: test_batch, sphericity: sphericityt, margin: margint, lobulation: lobulationt,
                                 spiculation: spiculationt, real_label: test_label, keep_prob: 1.0}  # use 1.0 as keep_prob for testing
                    acc_test, loss = sess.run([accruacy, net_loss], feed_dict=test_dict)

                    log('accuracy is %f, loss is %f, epoch %d time, consumed %.2f seconds' %
                        (acc_test, loss, i, (epoch_end - epoch_start)))
                    # estimate end time
                    end_time = datetime.datetime.now() + datetime.timedelta(seconds=(epoch_end - epoch_start) * (self.epoch - i - 1))
                    print('will end at %s' % end_time.strftime('%H:%M:%S'))

                    if acc_test > highest_acc:
                        highest_acc = acc_test
                        highest_iterator = i
                        saver.save(sess, dir.ckpt_path + 'ckpt', global_step=i + 1)
            log('training finshed. highest accuracy is %f, the iterator is %d ' % (highest_acc, highest_iterator))
            logtime('end time: ')
        else:
            # softmax layer
            real_label = tf.placeholder(tf.float32, [None, 2])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label)
            # cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
            net_loss = tf.reduce_mean(cross_entropy)

            prediction = tf.nn.softmax(net_out)
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))

            accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # allow memory allocation growth
            config = tf.ConfigProto()
            # config.gpu_options.per_process_gpu_memory_fraction = 0.4
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                # saver = tf.train.import_meta_graph('ckpt/archi-1-40.meta')
                # saver.restore("/ckpt/archi-1-40.data-00000-of-00001")
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                sess.run(tf.global_variables_initializer())
                save_path = tf.train.latest_checkpoint(dir.ckpt_path)
                meta_path = save_path + '.meta'
                log("restored from: %s" % save_path)
                saver = tf.train.import_meta_graph(meta_path)
                saver.restore(sess, save_path)
                # test_filenames = get_high_data(npy_path)
                # test_filenamesX = []
                # for onefile in test_filenames:
                #     if 'low' in onefile:
                #         test_filenamesX.append(onefile)
                #     if 'high' in onefile:
                #         test_filenamesX.append(onefile)

                # testnum = len(test_filenamesX) // self.epoch
                # print('test ', len(test_filenamesX))

                test_batch, sphericityt, margint, lobulationt, spiculationt, test_label = \
                    get_batch_withlabels(dir.npy_path, test_filenames)
                test_dict = {x: test_batch, sphericity: sphericityt, margin: margint, lobulation: lobulationt,
                             spiculation: spiculationt, real_label: test_label, keep_prob: 1.0}  # use 1.0 as keep_prob for testing

                acc_test, loss, aucpred = sess.run([accruacy, net_loss, prediction], feed_dict=test_dict)
                log('test accuracy is %f' % acc_test)

                acc_train_avg = 0
                for t in range(times):
                    batch_files = train_filenames[t * self.batch_size:(t + 1) * self.batch_size]

                    train_batch, sphericityt, margint, lobulationt, spiculationt, test_label = \
                        get_batch_withlabels(dir.npy_path, batch_files)
                    train_dict = {x: train_batch, sphericity: sphericityt, margin: margint, lobulation: lobulationt,
                                 spiculation: spiculationt, real_label: test_label, keep_prob: 1.0}  # use 1.0 as keep_prob for testing

                    acc_train, loss, aucpred = sess.run([accruacy, net_loss, prediction], feed_dict=train_dict)
                    # log('train accuracy is %f in batch %d' % (acc_train, t))
                    acc_train_avg += acc_train
                acc_train_avg /= times
                log('average train accuracy is %f' % acc_train_avg)
