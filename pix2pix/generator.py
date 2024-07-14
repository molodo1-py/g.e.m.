from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, LeakyReLU, Input, Concatenate
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.initializers import RandomNormal

conv_init = RandomNormal(mean = 0.0, stddev= 0.02)

def encoder_block(layer_in, n_filters: int, batchnorm = True):
  '''Аргументы:
              layer_in: вход слоя
              n_filters: кол-во ядер (int)
              batch_norm: будет ли BatchNormalization() (bool)
  '''
  global conv_init
  x = Conv2D(n_filters,(4,4), strides=(2,2), padding = 'same', kernel_initializer=conv_init)(layer_in)
  if batchnorm:
    x = BatchNormalization()(x, training = True)
  x = LeakyReLU(alpha=0.2)(x)
  return x

def decoder_block(layer_in, bypass, n_filters: int, dropout = True):
  '''Аргументы:
              layer_in: вход предыдущего слоя
              bypass: вход с выхода соответствующего слоя энкодера
              n_filters: количество фильтров (int)
              dropout: будет ли Dropout(0.5) (bool)
  '''
  global conv_init
  g = Conv2DTranspose(n_filters,(4,4), strides=(2,2), padding = "same", kernel_initializer=conv_init)(layer_in)
  g = BatchNormalization()(g, training = True)
  if dropout:
    g = Dropout(0.5)(g, training=True)
  g = Concatenate()([g, bypass])
  g = Activation('relu')(g)
  return g

def create_unet_generator(image_shape=(256,256,3)):
  '''
  Ф-ция создания генератора на базе U-net
  image_shape: размер подаваемых изображений
               по умолчанию (256,256,3)
  '''
  global conv_init
  in_image = Input(shape=image_shape)
  '''Энкодер'''
  e1 = encoder_block(in_image, 64, batchnorm = False)
  e2 = encoder_block(e1, 128)
  e3 = encoder_block(e2, 256)
  e4 = encoder_block(e3, 512)
  e5 = encoder_block(e4, 512)
  e6 = encoder_block(e5, 512)
  e7 = encoder_block(e6, 512)
  '''Бутылочное горлышко'''
  b = Conv2D(512, (4,4), strides=(2,2), padding = 'same', kernel_initializer=conv_init)(e7)
  b = Activation('relu')(b)
  '''Декодер'''
  d1 = decoder_block(b, e7, 512)
  d2 = decoder_block(d1, e6, 512)
  d3 = decoder_block(d2, e5, 512)
  d4 = decoder_block(d3, e4, 512, dropout=False)
  d5 = decoder_block(d4, e3, 256, dropout=False)
  d6 = decoder_block(d5, e2, 128, dropout=False)
  d7 = decoder_block(d6, e1, 64, dropout=False)
  '''Выходной слой'''
  g = Conv2DTranspose(3, (4,4), strides=(2,2), padding = "same", kernel_initializer=conv_init)(d7)
  out_image = Activation('tanh')(g)
  model = Model(in_image, out_image)
  return model