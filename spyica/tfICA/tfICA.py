import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .tools import TfMeanLayer, FastICALayer, TfPCALayer, CNN, \
    tf_sigmoid, d_tf_sigmoid, tf_cube, d_tf_cube


class TfFastICA:

    def __init__(self, train_batch, num_epoch=10, learning_rate=0.005, print_size=100,
                 batch_size=10000, beta1=0.9, beta2=0.999, adam_e=1e-8):

        self.train_batch = train_batch
        self._num_epoch = num_epoch
        self._learning_rate = learning_rate
        self._print_size = print_size
        self._batch_size = batch_size
        self._beta1 = beta1
        self._beta2 = beta2
        self._adam_e = adam_e

    def run(self):
        l1 = CNN(3, 1, 1, act=tf_sigmoid, d_act=d_tf_sigmoid)
        l2 = CNN(3, 1, 1, act=tf_sigmoid, d_act=d_tf_sigmoid)
        l3 = CNN(3, 1, 1, act=tf_sigmoid, d_act=d_tf_sigmoid)

        pca_l_1 = TfPCALayer(8)
        pca_l_2 = TfPCALayer(8)
        pca_l_3 = TfPCALayer(8)
        pca_l_4 = TfPCALayer(8)
        ica_l_1 = FastICALayer(8, 8, act=tf_cube, d_act=d_tf_cube)
        ica_l_2 = FastICALayer(8, 8, act=tf_cube, d_act=d_tf_cube)
        ica_l_3 = FastICALayer(8, 8, act=tf_cube, d_act=d_tf_cube)
        ica_l_4 = FastICALayer(8, 8, act=tf_cube, d_act=d_tf_cube)

        x = tf.compat.v1.placeholder(shape=[self._batch_size, 32, 32, 1], dtype=tf.float64)
        layer1 = l1.feedforward(x, padding='VALID')
        layer2 = l2.feedforward(layer1, padding='VALID')
        layer3 = l3.feedforward(layer2, padding='VALID')
        layer_flat = tf.reshape(layer3, [self._batch_size, -1])

        pca_layer_1 = pca_l_1.feedforward(layer_flat[:int(self._batch_size/4), :])
        pca_layer_2 = pca_l_2.feedforward(layer_flat[int(self._batch_size/4):int(self._batch_size/2), :])
        pca_layer_3 = pca_l_3.feedforward(layer_flat[int(self._batch_size/2):int(self._batch_size*(3/4)), :])
        pca_layer_4 = pca_l_4.feedforward(layer_flat[int(self._batch_size*(3/4)):, :])

        ica_layer_1 = ica_l_1.feedforward(pca_layer_1)
        ica_layer_2 = ica_l_2.feedforward(pca_layer_2)
        ica_layer_3 = ica_l_3.feedforward(pca_layer_3)
        ica_layer_4 = ica_l_4.feedforward(pca_layer_4)
        all_ica_section = ica_layer_1 + ica_layer_2 + ica_layer_3 + ica_layer_4

        grad_ica_1, grad_ica_up_1 = ica_l_1.backprop_ica()
        grad_ica_2, grad_ica_up_2 = ica_l_2.backprop_ica()
        grad_ica_3, grad_ica_up_3 = ica_l_3.backprop_ica()
        grad_ica_4, grad_ica_up_4 = ica_l_4.backprop_ica()

        grad_pca_1 = pca_l_1.backprop(grad_ica_1)
        grad_pca_2 = pca_l_2.backprop(grad_ica_2)
        grad_pca_3 = pca_l_3.backprop(grad_ica_3)
        grad_pca_4 = pca_l_4.backprop(grad_ica_4)

        grad_pca_reshape = tf.reshape(tf.concat([grad_pca_1, grad_pca_2, grad_pca_3, grad_pca_4], 0),
                                      [self._batch_size, 26, 26, 1])
        grad_3, grad_3_up = l3.backprop(grad_pca_reshape, padding='VALID')
        grad_2, grad_2_up = l2.backprop(grad_3, padding='VALID')
        grad_1, grad_1_up = l1.backprop(grad_2, padding='VALID')
        grad_up = grad_ica_up_1 + grad_ica_up_2 + grad_ica_up_3 + grad_ica_up_4

        sess = tf.compat.v1.InteractiveSession()
        sess.run(tf.compat.v1.global_variables_initializer())
        results_for_animation = []
        for it in range(self._num_epoch):
            for current_batch_index in range(0, self.train_batch.shape[0], self._batch_size):
                current_data = self.train_batch[current_batch_index:current_batch_index+self._batch_size, :]
                sess_results = sess.run([all_ica_section, grad_up,
                                         ica_layer_1, ica_layer_2, ica_layer_3, ica_layer_4], feed_dict={x: current_data})
                print('iter: ', it, 'mean: ', sess_results[0].mean())

                temp_np_array = sess_results[2]
                ica_data = sess_results[3:]
                for ica_x in ica_data:
                    temp_np_array = np.vstack((temp_np_array, ica_x))
                results_for_animation.append(temp_np_array)

        # get results from layers
        sess_results = sess.run([layer1, layer2, layer3], feed_dict={x: self.train_batch[:self._batch_size]})
        all_data_c = []
        for temp_data in sess_results:
            print(temp_data.shape)
            reshape_data = temp_data.reshape(self._batch_size, -1, 1)
            reshape_data[:, :, 0] = (reshape_data[:, :, 0] - reshape_data[:, :, 0].min(1)[:, np.newaxis])
            all_data_c.append(reshape_data)

        # get results from ica
        sess_results = sess.run([pca_layer_1, pca_layer_2, pca_layer_3, pca_layer_4,
                                 ica_layer_1, ica_layer_2, ica_layer_3, ica_layer_4], feed_dict={x: self.train_batch[: self._batch_size]})
        for temp in sess_results:
            print(temp.shape)

        all_data = []
        for s in sess_results:
            s_reshape = s.reshape(s.shape[0], 26, 26, 1)
            s_reshape_2 = s_reshape.reshape(s.shape[0], -1, 1)
            s_reshape_2[:, :, 0] = (s_reshape_2[:, :, 0] - s_reshape_2[:, :, 0].min(1)[:, np.newaxis]) / \
                                   (s_reshape_2[:, :, 0].max(1) - s_reshape_2[:, :, 0].min(1))[:, np.newaxis]
            print(s_reshape_2.min(1).sum(), s_reshape_2.max(1).sum())
            all_data.append(s_reshape_2)

        # shows
        fig = plt.figure(figsize=(30, 30))
        columns = 10
        rows = 10
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(np.squeeze(self.train_batch[:self._batch_size][i - 1]), cmap='gray')
            plt.axis('off')
            plt.title(str(i))
        plt.show()
        print('-------------------------------------')

        for temp in all_data_c:
            fig = plt.figure(figsize=(30, 30))
            columns = 10
            rows = 10
            for i in range(1, columns * rows + 1):
                fig.add_subplot(rows, columns, i)
                try:
                    num = int(np.sqrt(temp[i - 1].shape[0]))
                    plt.imshow(np.squeeze(temp[i - 1]).reshape(num, num), cmap='gray')
                except:
                    break
                plt.axis('off')
                plt.title(str(i))
            plt.show()
            print('-------------------------------------')

        count = 0
        for temp in all_data:
            fig = plt.figure(figsize=(30, 30))
            columns = 10
            rows = 10
            for i in range(1, columns * rows + 1):
                try:
                    num = int(np.sqrt(temp[i - 1].shape[0]))
                    fig.add_subplot(rows, columns, i)
                    plt.imshow(np.squeeze(temp[i - 1]).reshape(num, num), cmap='gray')
                    plt.axis('off')
                    plt.title(str(i))
                except:
                    break
            count = count + 1
            if count == 5:
                print('-------------------------------------')
            plt.show()
