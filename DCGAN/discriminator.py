from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, LeakyReLU, Dropout, Flatten

def discriminator_model():
    '''
    Модель дискриминатора, на вход ничего не подаём
    '''
    init_weights_kernel = RandomNormal(mean=0.0, stddev=0.02)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64, 64, 3), padding='same', kernel_initializer=init_weights_kernel, use_bias=False))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=init_weights_kernel, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=init_weights_kernel, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer=init_weights_kernel, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model