import random
from pix2pix.utils import minibatch
import matplotlib.pyplot as plt
from pix2pix.utils import dataset_separator
import time
import numpy as np


def img_transform_predict_compare(model, cnt_img_show, file_list,
                                  image_size = 300, direction = 0):
  '''Аргументы:
              model - модель генератора
              cnt_img_show - кол-во показываемых изображений
              file_list - список файлов (имя/путь)
              direction - порядок трансформации
  '''
  random.shuffle(file_list)
  dataA, dataB = minibatch(file_list, cnt_img_show, 0, image_size,
                            direction)
  a2b_img = model.predict(dataA)

  plt.figure(figsize=(18,9))
  cnter = 0
  for i in range(cnt_img_show):
    '''Счётчик прогресса'''
    cnter += 1
    print(f'\r__{(i/cnt_img_show)*100}%__', end = "", flush = True)
    '''Исходное изображение'''
    ax = plt.subplot(3, cnt_img_show, i+1)
    img_a = dataA[i]
    plt.imshow(img_a*0.5 + 0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    '''Трансформированное изображение'''
    ax=plt.subplot(3, cnt_img_show, i+1)
    plt.imshow(a2b_img[i]*0.5 + 0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    '''Оригинал'''
    ax = plt.subplot(3, cnt_img_show, i+1)
    img_b = dataB[i]
    plt.imshow(img_b*0.5 + 0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()


def print_log_progress(startTime, current, amount, params, epoch): #Cломан
  '''
  Прогрессбар как в керасе(keras)
  Вроде такого: 20/100 [====>----------------]

  Аргументы:
            startTime: время начала эпохи
            current: номер текущей операции
            ammount: общая длина цикла
            params: можно передать параметры (loss и т.п.)
            epoch: кол-во эпох
  '''
  bar_len = 30
  percent = int(current * bar_len/amount)
  progressbar = ""
  for i in range(bar_len):
    if i < percent:
      progressbar += "="
    elif i == percent:
      progressbar += ">"
    else:
      progressbar += "-"
  percent = round(((current/amount)*100), 2)
  time2stop = round(((time.time() - startTime)/current * amount)-((time.time() - startTime)), 2)

  message = "\r" + "Эпоха: " + str(epoch+1) + ' ' + str(current) + '/' + str(amount) + \
  f' [{progressbar}] ' + str(percent) + f'% , Времени - Осталось: {str(round(time2stop/60, 2))} мин,' \
  + f' Прошло: {(round(time.time() - startTime)/60, 2)} мин '

  for key in params:
    message += key + str(params[key]) + '. '

  print(message, end='') 



def take_real_batch(dataset, patch_shape):
  '''
  Ф-ция формирования пакета с реальными метками

  Аргументы:
          dataset: набор данных (картинок)
          patch_shape: размер выходных данных из дискриминатора
                      (нужно, чтобы сгенерировать необходимое кол-во меток)
  '''
  train_A, train_B = dataset
  labels_real = np.ones((len(train_A), patch_shape[0], patch_shape[1], 1))
  return [train_A, train_B], labels_real

def generate_fake_batch(gen_model, img_set_A, patch_shape):
  '''
  Ф-ция формирования пакета с фейковыми метками

  Аргументы:
          gen_model: модель генератор
          img_set_A: набор исходных изображений
          patch_shape: размер выходных данных из дискриминатора
                      (нужно, чтобы сгенерировать необходимое кол-во меток)
  '''
  syn_b = gen_model.predict(img_set_A)
  labels_fake = np.zeros((len(syn_b), patch_shape[0], patch_shape[1], 1))
  return syn_b, labels_fake


def train(dis_model, gen_model, Pix2Pix_model,
          file_list, n_epochs = 100, batch_size = 5,
          image_size = 256, direction = 0, directory = '/content/'):
  '''
  Ф-ция тренировки модели

  Аргументы:
          dis_model: модель дискриминатора
          gen_model: модель генератора
          Pix2Pix_model: модель Pix2Pix (GAN)
          file_list: список файлов
          n_epochs: кол-во эпох, по-умолчанию 100
          batch_size: размер батча (пакета изображений), по умолчанию 5
          image_size: высота изпбражения (т.к. изображения в датасете склеены
                      задаём только высоту) по умолчанию 256
          direction: порядок трансформации, по умолчанию 0
          directory: директория, куда будут сохраняться модели
  '''

  n_patch = dis_model.output_shape[1:]
  train_AB, val_AB = dataset_separator(file_list)
  startTime = time.time()
  batch_per_epoch = len(train_AB)
  random.shuffle(file_list)
  '''Обучение'''
  for epoch in range(n_epochs):
    for i in range(0, batch_per_epoch, batch_size):

      dataset = minibatch(train_AB, batch_size, i, image_size, direction)
      [x_real_A, x_real_B], real_labels = take_real_batch(dataset, n_patch)
      x_fake_B, x_fake_labels = generate_fake_batch(gen_model, dataset[0], n_patch)

      d_loss1 = dis_model.train_on_batch([x_real_A, x_real_B], real_labels)
      d_loss2 = dis_model.train_on_batch([x_real_A, x_fake_B], x_fake_labels)
      g_loss, _ , _ = Pix2Pix_model.train_on_batch(x_real_A, [real_labels, x_real_B])

      params = {
          'discr_real_loss: ': round(d_loss1, 3),
          'discr_fake_loss: ': round(d_loss2, 3),
          'Gen_loss: ': round(g_loss, 3)
          }
      print_log_progress(startTime, i+1, batch_per_epoch, params, epoch)
    '''Сохранение моделей (каждую эпоху)'''
    dis_model.save(directory + 'dis_model.h5')
    gen_model.save(directory + 'gen_model.h5')

    img_transform_predict_compare(gen_model, 5, val_AB, image_size, direction = direction)
    print()