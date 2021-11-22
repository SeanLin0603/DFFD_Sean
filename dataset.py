from imageio import imread
import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# DATABASE = 'D:\\Dataset\\DFFD\\data\\'
DATABASE = 'C:\\Users\\SeanVIP\\Documents\\FaceForensic\\align\\'

DATASETS = {
  'Real': 0,
  'Fake': 1
  }

class DatasetInstance:
  def __init__(self, label_name, label, datatype, img_size, map_size, norm, seed, bs, drop_last):
    self.img_size = img_size
    self.map_size = map_size

    
    self.transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.Normalize(*norm)
    ])
    
    self.transform_mask = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(map_size),
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor()
    ])
  
    self.label_name = label_name
    self.label = label
    self.datatype = datatype
    self.data_dir = '{0}{1}/{2}/'.format(DATABASE, self.datatype, self.label_name)
    files = os.listdir(self.data_dir)
    if self.datatype != 'test':
      random.Random(seed).shuffle(files)
    self.images = ['{0}/{1}'.format(self.data_dir, _) for _ in files]
    self.loader = DataLoader(self, num_workers=8, batch_size=bs, shuffle=(self.datatype != 'test'), drop_last=drop_last, pin_memory=True)
    
    print('Constructed Dataset `{0}` of size `{1}`'.format(self.data_dir, self.__len__()))

  def load_image(self, path):
    return self.transform(imread(path))

  def load_mask(self, path):
    return self.transform_mask(imread(path))

  def __getitem__(self, index):
    im_name = self.images[index]
    img = self.load_image(im_name)
    if self.label_name == 'Real':
      msk = torch.zeros(1,19,19)
    else:
      msk = self.load_mask(im_name.replace('Fake/', 'Mask/'))
    return {'img': img, 'msk': msk, 'lab': self.label, 'im_name': im_name}

  def __len__(self):
    return len(self.images)

class Dataset:
  def __init__(self, datatype, bs, img_size, map_size, norm, seed):
    drop_last = datatype == 'train'
    datasets = [DatasetInstance(_, DATASETS[_], datatype, img_size, map_size, norm, seed, bs, drop_last) for _ in DATASETS]
    drop_last = datatype == 'train' or datatype == 'eval'
    self.datasets = datasets


