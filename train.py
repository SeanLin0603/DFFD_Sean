import numpy as np
import os
import random
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn

from dataset import Dataset
from templates import get_templates

MODEL_DIR = './models/'
BACKBONE = 'xcp'
MAPTYPE = 'tmp'
BATCH_SIZE = 10
MAX_EPOCHS = 1
STEPS_PER_EPOCH = 630
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001

CONFIGS = {
  'xcp': {
          'img_size': (299, 299),
          'map_size': (19, 19),
          'norms': [[0.5] * 3, [0.5] * 3]
         },
  'vgg': {
          'img_size': (299, 299),
          'map_size': (19, 19),
          'norms': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
         }
}
CONFIG = CONFIGS[BACKBONE]



def calculate_losses(batch):
  img = batch['img']
  msk = batch['msk']
  lab = batch['lab']
  x, mask, vec = MODEL.model(img)
  loss_l1 = LOSS_L1(mask, msk)
  loss_cse = LOSS_CSE(x, lab)
  loss = loss_l1 + loss_cse
  pred = torch.max(x, dim=1)[1]
  acc = (pred == lab).float().mean()
  return { 'loss': loss, 'loss_l1': loss_l1, 'loss_cse': loss_cse, 'acc': acc }

def process_batch(batch, mode):
  if mode == 'train':
    MODEL.model.train()
    losses = calculate_losses(batch)
    OPTIM.zero_grad()
    losses['loss'].backward()
    OPTIM.step()
  elif mode == 'eval':
    MODEL.model.eval()
    with torch.no_grad():
      losses = calculate_losses(batch)
  return losses

SUMMARY_WRITER = SummaryWriter(MODEL_DIR + 'logs/')
def write_tfboard(item, itr, name):
  SUMMARY_WRITER.add_scalar('{0}'.format(name), item, itr)

# def run_step(e, s):
#   batch = DATA_TRAIN.get_batch()
#   losses = process_batch(batch, 'train')

#   if s % 10 == 0:
#     print('\r{0} - '.format(s) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
#   if s % 100 == 0:
#     print('\n', end='')
#     [write_tfboard(losses[_], e * STEPS_PER_EPOCH + s, _) for _ in losses]

def run_epoch(e):
  # for s in range(STEPS_PER_EPOCH):
  #   run_step(e, s)

  print('Epoch: {0}'.format(e))
  step = 0

  # Train
  realTrainLoader = DATA_TRAIN.datasets[0].loader
  fakeTrainLoader = DATA_TRAIN.datasets[1].loader
  for batch in zip(realTrainLoader, fakeTrainLoader):
    batch = list(batch)
    # print("[Info] Real image: {}".format(batch[0]['img'].shape))
    # print("[Info] Real mask: {}".format(batch[0]['msk'].shape))
    # print("[Info] Real label: {}".format(batch[0]['lab']))
    # print("[Info] Real imName: {}".format(batch[0]['im_name']))
    # print("[Info] Fake image: {}".format(batch[1]['img'].shape))
    # print("[Info] Fake mask: {}".format(batch[1]['msk'].shape))
    # print("[Info] Fake label: {}".format(batch[1]['lab']))
    # print("[Info] Fake imName: {}".format(batch[1]['im_name']))
    # print("\n")

    img = torch.cat([_['img'] for _ in batch], dim=0).cuda()
    msk = torch.cat([_['msk'] for _ in batch], dim=0).cuda()
    lab = torch.cat([_['lab'] for _ in batch], dim=0).cuda()
    #im_name = torch.cat([_['im_name'] for _ in batch], dim=0)
    input = { 'img': img, 'msk': msk, 'lab': lab }
    # print(input)

    losses = process_batch(input, 'train')

    if step % 10 == 0:
      print('\r{0} - '.format(step) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
    if step % 100 == 0:
      print('\n', end='')
      [write_tfboard(losses[_], e * STEPS_PER_EPOCH + step, _) for _ in losses]
    
    step = step + 1

  # # Eval
  # realEvalLoader = DATA_EVAL.datasets[0].loader
  # fakeEvalLoader = DATA_EVAL.datasets[1].loader
  # for batch in zip(realEvalLoader, fakeEvalLoader):
  #   batch = list(batch)
    
  #   img = torch.cat([_['img'] for _ in batch], dim=0).cuda()
  #   msk = torch.cat([_['msk'] for _ in batch], dim=0).cuda()
  #   lab = torch.cat([_['lab'] for _ in batch], dim=0).cuda()
  #   #im_name = torch.cat([_['im_name'] for _ in batch], dim=0)
  #   input = { 'img': img, 'msk': msk, 'lab': lab }
  #   # print(input)

  #   losses = process_batch(input, 'eval')
  #   print('[Info] Evaluation loss: {}'.format(losses))

  MODEL.save(e+1, OPTIM, MODEL_DIR)

if __name__ == "__main__":

  if BACKBONE == 'xcp':
    from xception import Model
    # from xception_new import Model
  elif BACKBONE == 'vgg':
    from vgg import Model

  torch.backends.deterministic = True
  SEED = 1
  random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)

  print("[Info] Build dataset")
  DATA_TRAIN = Dataset('train', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)
  DATA_EVAL = Dataset('eval', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)
  print("[Info] Built dataset\n\n")

  TEMPLATES = None
  if MAPTYPE in ['tmp', 'pca_tmp']:
    TEMPLATES = get_templates()

  MODEL_NAME = '{0}_{1}'.format(BACKBONE, MAPTYPE)
  MODEL_DIR = MODEL_DIR + MODEL_NAME + '/'

  MODEL = Model(MAPTYPE, TEMPLATES, 2, False)

  OPTIM = optim.Adam(MODEL.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  MODEL.model.cuda()
  LOSS_CSE = nn.CrossEntropyLoss().cuda()
  LOSS_L1 = nn.L1Loss().cuda()
  MAXPOOL = nn.MaxPool2d(19).cuda()

  LAST_EPOCH = 0
  for e in range(LAST_EPOCH, MAX_EPOCHS):
    run_epoch(e)

