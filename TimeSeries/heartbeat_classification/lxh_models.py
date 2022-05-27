import tensorflow as tf
from tensorflow.keras import Sequential, utils, regularizers, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout, AvgPool1D, BatchNormalization, Activation, ConvLSTM2D

class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv1D(filters, 3, strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv1D(filters, 3, strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行
        # ，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv1D(filters, 1, strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()
        
        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv1D(self.out_filters, 3, strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling1D()
        self.f1 = tf.keras.layers.Dense(4, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y



class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.c1 = Conv1D(filters=64, kernel_size=3, padding='same')  # 卷积层1
        self.b1 = BatchNormalization()  # BN层1
        self.a1 = Activation('relu')  # 激活层1
        self.c2 = Conv1D(filters=64, kernel_size=3, padding='same')
        self.b2 = BatchNormalization()  # BN层1
        self.a2 = Activation('relu')  # 激活层1
        self.p1 = MaxPool1D(pool_size=2, strides=2, padding='same')


        self.c3 = Conv1D(filters=128, kernel_size=3, padding='same')
        self.b3 = BatchNormalization()  # BN层1
        self.a3 = Activation('relu')  # 激活层1
        self.c4 = Conv1D(filters=128, kernel_size=3, padding='same')
        self.b4 = BatchNormalization()  # BN层1
        self.a4 = Activation('relu')  # 激活层1
        self.p2 = MaxPool1D(pool_size=2, strides=2, padding='same')
 

        self.c5 = Conv1D(filters=256, kernel_size=3, padding='same')
        self.b5 = BatchNormalization()  # BN层1
        self.a5 = Activation('relu')  # 激活层1
        self.c6 = Conv1D(filters=256, kernel_size=3, padding='same')
        self.b6 = BatchNormalization()  # BN层1
        self.a6 = Activation('relu')  # 激活层1
        self.c7 = Conv1D(filters=256, kernel_size=3, padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p3 = MaxPool1D(pool_size=2, strides=2, padding='same')


        self.c8 = Conv1D(filters=512, kernel_size=3, padding='same')
        self.b8 = BatchNormalization()  # BN层1
        self.a8 = Activation('relu')  # 激活层1
        self.c9 = Conv1D(filters=512, kernel_size=3, padding='same')
        self.b9 = BatchNormalization()  # BN层1
        self.a9 = Activation('relu')  # 激活层1
        self.c10 = Conv1D(filters=512, kernel_size=3, padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p4 = MaxPool1D(pool_size=2, strides=2, padding='same')


        self.c11 = Conv1D(filters=512, kernel_size=3, padding='same')
        self.b11 = BatchNormalization()  # BN层1
        self.a11 = Activation('relu')  # 激活层1
        self.c12 = Conv1D(filters=512, kernel_size=3, padding='same')
        self.b12 = BatchNormalization()  # BN层1
        self.a12 = Activation('relu')  # 激活层1
        self.c13 = Conv1D(filters=512, kernel_size=3, padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p5 = MaxPool1D(pool_size=2, strides=2, padding='same')

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(512, activation='relu')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(4, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)


        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)


        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)


        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)


        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)


        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y
    
class DilationNet(Model):
    def __init__(self):
        super(DilationNet, self).__init__()
        self.conv1 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape = (205, 1))
        self.conv2 = Conv1D(filters=32, kernel_size=3, dilation_rate=2, padding='same', activation='relu')
        self.conv3 = Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='same', activation='relu')
        self.conv4 = Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool1 = MaxPool1D(pool_size=3, strides=2, padding='same')
        
        self.conv5 = Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.conv6 = Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool2 = MaxPool1D(pool_size=3, strides=2, padding='same')
        
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        
        self.fc1 = Dense(units=256, activation='relu')
        self.fc21 = Dense(units=16, activation='relu')
        self.fc22 = Dense(units=256, activation='sigmoid')
        self.fc3 = Dense(units=4, activation='softmax')
            
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        
        x = self.conv5(x)
        x = self.conv6(x) 
        x = self.max_pool2(x)
        
        x = self.dropout(x)
        x = self.flatten(x)
        
        x1 = self.fc1(x)
        x2 = self.fc22(self.fc21(x))
        x = self.fc3(x1+x2)
        
        return x 
    

def simple_LSTM():
    return tf.keras.Sequential([
              LSTM(32, return_sequences=True,input_shape=(205, 1)),
              LSTM(32, return_sequences=True),
              LSTM(32),
              Dense(4, activation='softmax')])
def CNN_LSTM():
    return tf.keras.Sequential([
    TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, 41, 1)),
    TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')),
    TimeDistributed(Dropout(0.5)),
    TimeDistributed(MaxPool1D(pool_size=2)),
    TimeDistributed(Flatten()),
    LSTM(100),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dense(4, activation='softmax')])

def ConvLSTM():
    return tf.keras.Sequential([
    ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(5, 1, 41, 1)),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(4, activation='softmax')])