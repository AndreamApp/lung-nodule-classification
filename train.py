# -*- coding:utf-8 -*-
import tensorflow as tf
import time
import datetime
import os
import tools
from tools import log, logtime, count_high, count_low
from selnet import Sel3DCNNConfig, Sel3DCNN

def train(base_dir):
    # allow memory allocation growth
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        config = Sel3DCNNConfig()
        selcnn = Sel3DCNN(config)
        selcnn.setup_graph()

        # checkpoints
        dir = tools.workspace(base_dir)
        saver = tf.train.Saver(max_to_keep=selcnn.epoch_time)

        # tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(dir.tensorboard_path, sess.graph)

        # start training...

        tools.open_log_file('log/selcnn.txt')
        log('input npy dir: %s' % dir.npy_path)
        log('cubic shape: (%d, %d, %d)' % (selcnn.cubic_shape[0], selcnn.cubic_shape[1], selcnn.cubic_shape[2]))
        log('epoch: %d' % selcnn.epoch_time)
        log('batch size: %d' % selcnn.batch_size)
        log('keep prob: %f' % selcnn.keep_prob)
        logtime('start time: ')

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.01, global_step, selcnn.batch_time * 40, 1, staircase=True)
        selnet_train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(selcnn.selnet_loss)
        clsnet_train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(selcnn.clsnet_loss)

        sess.run(tf.global_variables_initializer())

        # train selnet
        print('preparing selnet data')
        selcnn.prepare_selnet_data(base_dir)
        for epoch in range(selcnn.epoch_time):
            epoch_start = time.time()

            print(f'----- epoch {epoch} -----')
            selcnn.next_epoch()
            for batch in range(selcnn.batch_time):
                print('\r', ('%d/%d' % (batch+1, selcnn.batch_time)).ljust(10), end='', flush=True)
                batch_x, batch_y = selcnn.next_batch()
                feed_dict = {selcnn.input_x: batch_x, selcnn.selnet_label: batch_y}
                _, summary = sess.run([selnet_train_step, merged], feed_dict=feed_dict)
                lnrt = sess.run(learning_rate, feed_dict={global_step: batch + epoch * selcnn.batch_time})
                train_writer.add_summary(summary, batch)
            
            test_x, test_y = selcnn.test_batch()
            feed_dict = {selcnn.input_x: test_x, selcnn.selnet_label: test_y}
            pred, acc, loss = sess.run([selcnn.selnet_prediction, selcnn.selnet_accuracy, selcnn.selnet_loss], feed_dict=feed_dict)

            epoch_end = time.time()
            log('accuracy is %f, loss is %f, epoch %d time, consumed %.2f seconds' %
                (acc, loss, epoch, (epoch_end - epoch_start)))
            # estimate end time
            end_time = datetime.datetime.now() + datetime.timedelta(seconds=(epoch_end - epoch_start) * (selcnn.epoch_time - epoch - 1))
            print('will end at %s' % end_time.strftime('%H:%M:%S'))
            saver.save(sess, dir.ckpt_path + 'selnet', global_step=epoch)

        # TODO: freeze selnet
        
        # train clsnet

        print('preparing clsnet data')
        selcnn.prepare_clsnet_data(base_dir)
        for epoch in range(selcnn.epoch_time):
            epoch_start = time.time()

            print(f'----- epoch {epoch} -----')
            selcnn.next_epoch()
            for batch in range(selcnn.batch_time):
                print('\r', ('%d/%d' % (batch+1, selcnn.batch_time)).ljust(10), end='', flush=True)
                batch_x, batch_y = selcnn.next_batch()
                feed_dict = {selcnn.input_x: batch_x, selcnn.clsnet_label: batch_y}
                _, summary = sess.run([clsnet_train_step, merged], feed_dict=feed_dict)
                lnrt = sess.run(learning_rate, feed_dict={global_step: batch + epoch * selcnn.batch_time})
                train_writer.add_summary(summary, batch)
            
            test_x, test_y = selcnn.test_batch()
            feed_dict = {selcnn.input_x: test_x, selcnn.clsnet_label: test_y}
            pred, acc, loss = sess.run([selcnn.clsnet_prediction, selcnn.clsnet_accuracy, selcnn.clsnet_loss], feed_dict=feed_dict)

            epoch_end = time.time()
            log('accuracy is %f, loss is %f, epoch %d time, consumed %.2f seconds' %
                (acc, loss, epoch, (epoch_end - epoch_start)))
            # estimate end time
            end_time = datetime.datetime.now() + datetime.timedelta(seconds=(epoch_end - epoch_start) * (self.epoch - i - 1))
            print('will end at %s' % end_time.strftime('%H:%M:%S'))
            saver.save(sess, dir.ckpt_path + 'clsnet', global_step=epoch)
        

if __name__ == '__main__':
    train('data')
