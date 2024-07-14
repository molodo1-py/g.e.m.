import numpy as np
from PIL import Image
import glob


def get_file_paths(tamplate):
  '''
  Для получения file_list(для использования следующих ф-ций)
  tamplate - шаблон поиска
  '''
  return glob.glob(tamplate, recursive=True)


def read_image(file_name, image_size = 300, direction = 0):
  ''' 
  Использовать, если в датасете изображения склеены
  Аргументы:
        file_name - имя файла
        image_size - размер изображения, по умолчанию 600x300
        direction: порядок изображений: 0 = A --> B 
                             любое другое = B --> A
            
  '''
  
  '''Подгоняем, нормируем и разрезаем изображения'''
  img = Image.open(file_name)
  img = img.resize((image_size*2, image_size), Image.BILINEAR)
  arr = np.array(img)/127.5-1
  imgA = arr[:, image_size:, :]
  imgB = arr[:, :image_size, :]
  '''Возврат в порядке заданном direction'''
  if direction == 0:
    return imgA, imgB
  return imgB, imgA

def dataset_separator(file_list):
  '''
  Формирование списков путей тренировочной и проверочной выборки
  file_list - список файлов с путями
  '''
  train_AB = []
  val_AB = []
  for i in range(len(file_list)):
    if 'val' in file_list[i]:
      val_AB.append(file_list[i])
    else:
      train_AB.append(file_list[i])
  return train_AB, val_AB

def minibatch(file_list, batch_size, iteration = 0, image_size = 300, direction = 0):
  '''
  Формирование мини-пакетов изображений 
  Аргументы:
        file_list - список файлов
        batch_size - размер батча(пакета изображений)
        iteration - текущий индекс в списке файлов
        direction - порядок трансформации
  '''
  dataA = []
  dataB = []
  curr_list = file_list[iteration:iteration + batch_size]
  for sample in curr_list:
    imgA, imgB = read_image(sample, image_size, direction)
    dataA.append(imgA)
    dataB.append(imgB)
  print(dataA[0].shape)
  dataA = np.float32(dataA)
  dataB = np.float32(dataB)
  return dataA, dataB