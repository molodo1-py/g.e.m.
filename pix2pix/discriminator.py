from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

conv_init = RandomNormal(mean = 0.0, stddev= 0.02)
gamma_init = RandomNormal(1., 0.02)

def CBL(layer, filters: int, in_layer = False,
        out_layer = False,
        kernel_size = (4,4),
        strides = (2,2),
        padding = 'same'):
  '''Аргументы:
              layer: вход слоя
              filters: кол-во ядер (int)
              in_layer: есть ли входной слой (bool), по умолчанию True
              out_layer: есть ли входной слой (bool), по умолчанию False
              kernel_size: размер ядра (int, int)
              strides: шаг (int, int)
              padding: "same"/"valid" заполненность
  '''
  x = Conv2D(filters, kernel_size, padding = padding, kernel_initializer=conv_init)(layer)
  if not in_layer:
    x = BatchNormalization(momentum=0.9, epsilon=1.01e-5, gamma_initializer = gamma_init)(x)
  x = LeakyReLU(alpha=0.2)(x)
  return x

def create_discriminator(img_shape):
  '''
  Ф-ция создания Дискриминатора
  
  img_shape: размер подаваемых изображений
  '''
  
  in_source_img = Input(shape=img_shape) 
  in_target_img = Input(shape=img_shape)
  merged = Concatenate()([in_source_img, in_target_img])
  d = CBL(merged, 64, in_layer=True)
  d = CBL(d, 128)
  d = CBL(d, 256)
  d = CBL(d, 512)
  d = CBL(d, 512, out_layer=True)
  out_dist = Conv2D(1, (4,4), padding = "same", kernel_initializer=conv_init, activation="sigmoid")(d)
  model = Model([in_source_img, in_target_img], out_dist)
  '''Компиляция'''
  model.compile(loss='binary_crossentropy', optimizer = Adam(learning_rate=0.0002, beta_1=0.5),
                loss_weights=[0.5])
  return model