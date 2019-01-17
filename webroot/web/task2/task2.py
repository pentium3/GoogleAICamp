""" Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
from scipy.spatial.distance import cdist
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
def conv_block(input_layer, mode, filters=64, dropout=0.0):
    conv = tf.layers.conv2d(inputs=input_layer,
                            filters=filters,
                            kernel_size=[5, 5],
                            padding="same",
                            activation=tf.nn.relu)
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

    dropout_layer = tf.layers.dropout(inputs=pool, rate=dropout, training=mode)

    return dropout_layer


class ConvNet(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 4
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.n_test = 10000
        self.training=False

    # def get_pred_data(self):
    #     with tf.name_scope('data'):
    #         imgset = []
    #         labset = []
    #         img = cv2.imread("./train/data/000005.jpg")
    #         img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
    #         lab = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    #         for i in range(self.batch_size):
    #             imgset.append(img)
    #             labset.append(lab)
    #         imgset=np.array(imgset).astype(np.float32)
    #         labset=np.array(labset).astype(np.float32)
    #         print(imgset.shape, labset.shape)
    #
    #         pred = (imgset, labset)
    #         pred_data = tf.data.Dataset.from_tensor_slices(pred)
    #         pred_data = pred_data.batch(self.batch_size)
    #         iterator = tf.data.Iterator.from_structure(pred_data.output_types, pred_data.output_shapes)
    #         #img, self.label = iterator.get_next()
    #         #self.img = tf.reshape(img, shape=[-1, 224, 224, 3])
    #         self.pred_init = iterator.make_initializer(pred_data)
    #         print("pred_init inited")

    def get_data(self):
        with tf.name_scope('data'):
            # train_data, test_data = utils.get_mnist_dataset(self.batch_size)
            # train, val, test = utils.read_mnist('data/mnist', flatten=False)
            inp = pd.read_csv('./train/verify0.csv')
            df = pd.DataFrame(inp)
            image = []
            #            labels = []
            labels = np.zeros((len(df),10,2))
            name = ["black","blond","brown","glasses","gray","male","mouth","eye","nose","hair"]
            for i in range(0,len(df)):
                loc = []
                for j in range(len(df.iloc[i])):
                    loc.append((df.iloc[i])[j])
                for j in range(1,len(loc)-1):
                    loc[j] = int(loc[j])
                img = cv2.imread("./train/data/" + loc[0])
                img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
                image.append(img)
                for j in range(1,len(loc)-1):
                    if loc[j] == 1:
                        labels[i][j-1][0] = 1
                    else:
                        labels[i][j-1][1] = 1

            print(len(image))

            inp1 = pd.read_csv('./train/test0.csv')
            df1 = pd.DataFrame(inp1)
            image1 = []
            labels1 = np.zeros((len(df1),10,2))
            for i in range(0,len(df1)):
                loc = []
                for j in range(len(df1.iloc[i])):
                    loc.append((df1.iloc[i])[j])
                for j in range(1,len(loc)-1):
                    loc[j] = int(loc[j])
                img = cv2.imread("./train/data/" + loc[0])
                img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
                for j in range(1,len(loc)-1):
                    if loc[j] == 1:
                        labels1[i][j-1][0] = 1
                    else:
                        labels1[i][j-1][1] = 1

                image1.append(img)

            image=np.array(image).astype(np.float32)
            labels=np.array(labels).astype(np.float32)
            image1=np.array(image1).astype(np.float32)
            labels1=np.array(labels1).astype(np.float32)

            imgset = []
            labset = []

            filename = "cs1048486361028912.jpg"
            img = cv2.imread("./cartoon/" + filename)
            img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
            lab = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
            for i in range(self.batch_size):
                imgset.append(img)
                labset.append(lab)


            imgset=np.array(imgset).astype(np.float32)
            labset=np.array(labset).astype(np.float32)
            print(imgset.shape, labset.shape)

            print(image.shape, labels.shape, image1.shape, labels1.shape)
            train = (image, labels)
            test = (image1, labels1)
            pred = (imgset, labset)

            train_data = tf.data.Dataset.from_tensor_slices(train)
            #train_data = train_data.shuffle(10000) # if you want to shuffle your data
            train_data = train_data.batch(self.batch_size)
            test_data = tf.data.Dataset.from_tensor_slices(test)
            test_data = test_data.batch(self.batch_size)
            pred_data = tf.data.Dataset.from_tensor_slices(pred)
            pred_data = pred_data.batch(self.batch_size)


            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 224, 224, 3])
            # self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data
            self.pred_init = iterator.make_initializer(pred_data)    # initializer for train_data


    def inference_by_layers(self, mode=False):
        '''
        Build the model according to the description we've shown in class
        Define the model by using tf.layers
        '''
        conv_layer = self.img
        filters = [32, 64, 128]
        dropout_rates = [0.2, 0.4, 0.7]
        for filter_num, dropout_rate in zip(filters, dropout_rates):
            conv_layer = conv_block(conv_layer, mode = mode,filters=filter_num, dropout=dropout_rate)
        pool4_flat = tf.layers.flatten(conv_layer)
        dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
        drop3 = tf.layers.dropout(inputs=dense, rate=0.4)
        logits = [0] * 10           # [10, 2]
        prediction = [0] * 10       # [10]

        black_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[0] = tf.layers.dense(inputs=black_dense, units=2)
        prediction[0] = tf.argmax(input=logits[0], axis=1)

        blond_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[1] = tf.layers.dense(inputs=blond_dense, units=2)
        prediction[1] = tf.argmax(input=logits[1], axis=1)

        brown_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[2] = tf.layers.dense(inputs=brown_dense, units=2)
        prediction[2] = tf.argmax(input=logits[2], axis=1)

        glasses_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[3] = tf.layers.dense(inputs=glasses_dense, units=2)
        prediction[3] = tf.argmax(input=logits[3], axis=1)

        gray_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[4] = tf.layers.dense(inputs=gray_dense, units=2)
        prediction[4] = tf.argmax(input=logits[4], axis=1)

        male_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[5] = tf.layers.dense(inputs=male_dense, units=2)
        prediction[5] = tf.argmax(input=logits[5], axis=1)

        mouth_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[6] = tf.layers.dense(inputs=mouth_dense, units=2)
        prediction[6] = tf.argmax(input=logits[6], axis=1)

        eye_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[7] = tf.layers.dense(inputs=eye_dense, units=2)
        prediction[7] = tf.argmax(input=logits[7], axis=1)

        nose_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[8] = tf.layers.dense(inputs=nose_dense, units=2)
        prediction[8] = tf.argmax(input=logits[8], axis=1)

        hair_dense = tf.layers.dense(inputs=drop3, units=1024)
        logits[9] = tf.layers.dense(inputs=hair_dense, units=2)
        prediction[9] = tf.argmax(input=logits[9], axis=1)


        self.logits = tf.cast(tf.convert_to_tensor(logits), tf.float32)
        print(self.logits[0].shape)
        self.logits=tf.stack([logits[0], logits[1], logits[2], logits[3], logits[4], logits[5], logits[6], logits[7], logits[8], logits[9]], axis=1)
        self.logits = tf.sigmoid(self.logits)

        # conv1 = tf.layers.conv2d(inputs=self.img, filters=32, kernel_size=[5, 5], padding='SAME', activation=tf.nn.relu, name='conv1')
        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
        # conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding='SAME', activation=tf.nn.relu, name='conv2')
        # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
        # 
        # feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        # pool2 = tf.reshape(pool2, [-1, feature_dim])
        # fc = tf.layers.dense(pool2, 1024, activation=tf.nn.relu, name='fc')
        # dropout = tf.layers.dropout(fc, tf.constant(0.75), training=self.training, name='dropout')
        # self.logits = tf.layers.dense(dropout, 10, name='logits')

    #        conv1 = tf.layers.conv2d(inputs=self.img, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    #        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #        drop1 = tf.layers.dropout(inputs=pool1, rate=0.2, training=self.training)
    #
    #        conv2 = tf.layers.conv2d(inputs=drop1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    #        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #        drop2 = tf.layers.dropout(inputs=pool2, rate=0.4, training=self.training)
    #
    #        conv3 = tf.layers.conv2d(inputs=drop2, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    #        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    #        drop3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)
    #
    #        feature_dim = drop3.shape[1] * drop3.shape[2] * drop3.shape[3]
    #        drop3 = tf.reshape(drop3, [-1, feature_dim])
    #        fc = tf.layers.dense(drop3, 1024, activation=tf.nn.relu, name='fc')
    #        dropout = tf.layers.dropout(fc, tf.constant(0.75), training=self.training, name='dropout')
    #        self.logits = tf.layers.dense(dropout, 10, name='logits')


    # self.logits=tf.cast(tf.convert_to_tensor(prediction), tf.float32)


    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        tf.nn.softmax_cross_entropy_with_logits
        softmax is applied internally
        don't forget to compute mean cross all sample in a batch
        '''
        with tf.name_scope('loss'):
            #tmp1 = tf.argmax(self.logits, 2)
            #logits = tf.placeholder(tf.float32,shape=(10,2))
            #tmp2 = tf.argmax(self.label, 2)
            #tmp1 = tf.cast(tmp1, tf.float32)
            #tmp2 = tf.cast(tmp2, tf.float32)
            loss = tf.reduce_sum(tf.nn.l2_loss(self.logits-self.label))
            #cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
            self.loss = loss

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        Don't forget to use global step
        '''
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
        self.opt = optimizer

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        Remember to track both training loss and test accuracy
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            #preds = tf.nn.softmax(self.logits)
            print(self.logits)
            print(self.label)
            print(tf.argmax(self.logits, 2))
            correct_preds = tf.equal(tf.argmax(self.logits, 2), tf.argmax(self.label, 2))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def predicting(self):
        with tf.name_scope('predict'):
            self.predictone = tf.argmax(self.logits, 2)
            # self.predictone = self.logits

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference_by_layers()
        self.loss()
        self.optimize()
        #self.get_pred_data()
        self.predicting()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % 20 == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                # train_writer.add_summary(summary, i)
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_starter/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        # utils.safe_mkdir('checkpoints')
        writer = tf.summary.FileWriter('./graphs/convnet_starter', tf.get_default_graph())
        merged = tf.summary.merge_all()


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            train_writer = tf.summary.FileWriter('checkpoints/checkpoint/train',
                                                 sess.graph)
            test_writer = tf.summary.FileWriter('checkpoints/checkpoint/test')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                print("epoch = ", epoch)
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)



            self.predict(sess)
        writer.close()

    def predict(self, sess):
        sess.run(self.pred_init)
        print("predict inited")
        ans = sess.run([self.predictone])
        print("ANS=====",ans[0])

        # dataset = pd.read_csv('./allDataset.csv')
        #
        # dataset=pd.DataFrame(dataset)
        # allData = []
        # for i in range(len(dataset)):
        #     oneImage = []
        #     for j in range(1,len(dataset.iloc[i])):
        #         oneImage.append(dataset.iloc[i][j])
        #
        #     allData.append(oneImage)
        #
        # print(allData)
        #
        # input = ans[0][0]
        # inputData = []
        # inputData.append(input)
        #
        # similar = 0
        # dis = 999999999
        # for i in range(len(allData)):
        #     compare = []
        #     compare.append(allData[i])
        #     distance = cdist(inputData,compare,metric='cosine')
        #     if distance < dis:
        #         dis = distance
        #         similar = i
        #
        # print(similar)

def task2(FILE, FILE2)
    model = ConvNet()
    model.build()
    model.train(n_epochs=1)
