from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

def create_Pix2Pix(g_model, d_model, image_shape=(256,256,3)):
  '''
  Аргументы:
          g_model: модель-генератор
          d_model: модель-дискриминатор
          image_shape: размер подаваемых изображений
                       по умолчанию (256,256,3)
  '''
  for layer in d_model.layers:
    if not isinstance(layer, BatchNormalization):
      layer.trainable = False
  '''Собираем модель'''
  in_source = Input(shape=image_shape)
  gen_out = g_model(in_source)
  dis_out = d_model([in_source, gen_out])
  model = Model(in_source, [dis_out, gen_out])
  opt = Adam(learning_rate=0.0002, beta_1=0.5)
  '''Компилируем'''
  model.compile(loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1,100])
  return model