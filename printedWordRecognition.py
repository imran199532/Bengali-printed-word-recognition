import keras
import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)

import os
from os.path import join
import json
import random
import itertools
import re
import datetime

import editdistance
import numpy as np
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,Dropout,BatchNormalization
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model, model_from_yaml
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import cv2

sess = tf.Session()
K.set_session(sess)

#direc = "E:/imran"

from collections import Counter
maxLen=15


def get_counter(dirpath):
    letters =set()

    for file in os.listdir(dirpath):
        filePath = 'C:/Users/Badhon/PycharmProjects/imran/modified_labels/%s' % file


        with open(filePath )as f:
            for line in f:
                letters.add(int(line))

    return Counter(letters)

path='C:/Users/Badhon/PycharmProjects/imran/modified_labels'
c_val = get_counter(path)


letters_test = set(c_val.keys())

letters = sorted(list(letters_test))
print(letters)



def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True


class TextImageGenerator:

    def __init__(self,
                 dirpath,
                 img_w, img_h,
                 batch_size,
                 downsample_factor,
                 max_text_len=maxLen
                 ):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor

        self.samples = []

        ################..------------------------------------------------>> changes has been made

        fin = 0

        for filename in os.listdir(dirpath):

            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                img_filepath = join(dirpath, filename)
                description =[]
                txt_filepath=join('modified_labels/%s.txt'%name)
                valid=1
                with open(txt_filepath)as f:
                    for line in f:
                        if int(line)>=74:
                            valid=0
                        description.append(int(line))


                if valid==0:
                    continue
                self.samples.append([img_filepath, description])
                #print(description)


        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)

    def get_output_size(self):
        return 74 + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def all_data(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.n, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.n, self.img_w, self.img_h, 1])
            #Y_data = np.ones([self.n])
            Y_data = np.ones([self.n, maxLen])
            input_length = np.ones((self.n, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.n, 1))
            source_str = []

            i=0
            for i in range(self.n):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img

                #Y_data[i] = text
                id = 0
                for val in text:
                    Y_data[i, id] = int(val)
                    id += 1
                for j in range(id, maxLen):
                    Y_data[i, j] = 0

                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                # 'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.n])}
            yield (inputs, outputs)
            break

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size,maxLen])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()


                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img

                textLen=len(text)
                id=0
                for val in text:
                    Y_data[i,id]=val
                    id+=1
                for j in range(id,maxLen):
                    Y_data[i,j]=0

                #Y_data[i] = text
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                # 'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)


##################----------------------->>>>>>>>>   END


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train(img_w, load=False):
    # Input Parameters
    print("Please wait for a while-----Training is in process!!!")

    img_h = 28

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    batch_size = 32
    downsample_factor = pool_size ** 2
    tiger_train = TextImageGenerator('train', img_w, img_h, batch_size, downsample_factor)
    tiger_train.build_data()

    tiger_val = TextImageGenerator(  'test', img_w, img_h, batch_size, downsample_factor)
    tiger_val.build_data()

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner=Dropout(0.2,name='drop1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = Dropout(0.2, name='drop2')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv3')(inner)
    inner = BatchNormalization()(inner)

    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))

    y_pred = Activation('softmax', name='softmax')(inner)



    Model(inputs=input_data, outputs=y_pred).summary()


    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
   
   
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

 

  
    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    if load:
        model = load_model('./tmp_model.h5', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)



    if not load:
        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        model.fit_generator(generator=tiger_train.next_batch(),
                            steps_per_epoch=tiger_train.n,
                            epochs=2,
                            validation_data=tiger_val.next_batch(),
                            validation_steps=tiger_val.n)



    return model


#####################---------------------- Loss and train functions, network architecture¶-----END

##################-----------------------Training the model



model = train(60, load=False)
# serialize model to YAML
model6_yaml = model.to_yaml()
with open("model6.yaml", "w") as yaml_file:
    yaml_file.write(model6_yaml)
# serialize weights to HDF5
model.save_weights("model6.h5")
print("Saved model to disk")

# later...

################------------------------------Function to decode neural network output

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(out):
    ret = []
    print('kkkl')
    #print(len(out[0]))



    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]

        ret.append(out_best)

    return ret




def longest_common_subsequence(A,B):
    dp=np.zeros((15,15))

    i=0
    j=0
    ans=0

    for p in A:
        p=int(p)
        if p==0:
            break
        j=0
        for q in B:
            q=int(q)
            if q==0:
                break
            if p==q:
                dp[i,j]=1
                if i-1>=0 and j-1>=0:
                    dp[i,j]+=dp[i-1,j-1]

            else:
                if i-1>=0:
                    dp[i,j]=dp[i-1,j]

                if j-1>=0 and dp[i,j-1]>dp[i-1,j]:
                    dp[i,j]=dp[i,j-1]

            j+=1

        i+=1
    return dp[i-1,j-1]



#########################------------------------END

#########################------------------------Test on validation images


tiger_test = TextImageGenerator('test', 60, 28, 32, 4)
tiger_test.build_data()

net_inp =model.get_layer(name='the_input').input
net_out =  model.get_layer(name='softmax').output

correct = 0
wrong = 0
total=0
count=0

for inp_value, _ in tiger_test.all_data():

    count+=1

    bs = inp_value['the_input'].shape[0]


    X_data = inp_value['the_input']

    net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})

    pred_texts=[]
    cnt=0
    for i in range(0,len(net_out_value)):
        temp=[]
        cnt=0
        for j in range(0,len(net_out_value[0])):
            mx=-1111
            best=1
            for k in range(0,75):
                if net_out_value[i,j,k]>mx:
                    mx=net_out_value[i,j,k]
                    best=k
            if best==74:
                continue

            if cnt==0:

                temp.append(best)
                cnt+=1
                continue

            temp.append(best)

        pred_texts.append(temp)


    labels = inp_value['the_labels']
    texts = []

    i=0
    j=0
    id=0

    for label in labels:
        text = label

        mx=0
        for val in text:
            if int(val)==0:
                break
            mx+=1


        ans=longest_common_subsequence(text,pred_texts[id])
        correct+=ans
        total+=mx

        id+=1

print("The accuracy on test data set is:")
print(correct*100 / total)



