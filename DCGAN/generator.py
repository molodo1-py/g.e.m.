from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2DTranspose, ReLU, Reshape

def generator_model():
    '''
    Модель генератора, т.к. используется в проекте
    ничего не подаём на вход
    '''
    init_weights_kernel = RandomNormal(mean=0.0, stddev=0.02)
    model = Sequential()
    model.add(Input(shape=(100,)))
    model.add(Dense(4 * 4 * 256))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, padding='same', strides=2, kernel_initializer=init_weights_kernel, use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(64, kernel_size=4, padding='same', strides=2, kernel_initializer=init_weights_kernel, use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(32, kernel_size=4, padding='same', strides=2, kernel_initializer=init_weights_kernel, use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(3, kernel_size=4, padding='same', strides=2, activation='tanh', kernel_initializer=init_weights_kernel, use_bias=False))
    return model