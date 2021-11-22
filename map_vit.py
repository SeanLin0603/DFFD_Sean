import numpy as np
import os
import random
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn

from config import Config as config
from dataset import Dataset
from templates import get_templates

if config.backbone == 'xcp':
  # from xception import Model
  from xception_map import Model
elif config.backbone == 'vgg':
  from vgg import Model


def write_tfboard(item, itr, name):
    summary_writer.add_scalar('{0}'.format(name), item, itr)

def calculate_losses(batch):
    img = batch['img']
    msk = batch['msk']
    mask = model.model(img)
    # print("[Info] mask: {}".format(msk))
    loss = lossL1Func(mask, msk)
    return { 'loss': loss}

def process_batch(batch, mode):
    if mode == 'train':
        model.model.train()
        losses = calculate_losses(batch)
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
    elif mode == 'eval':
        model.model.eval()
        with torch.no_grad():
            losses = calculate_losses(batch)
    return losses

def run_epoch(epoch):
    print("[Info] Epoch: {}".format(epoch))
    step = 0

    # Train
    realTrainLoader = trainData.datasets[0].loader
    fakeTrainLoader = trainData.datasets[1].loader
    for batch in zip(realTrainLoader, fakeTrainLoader):
        batch = list(batch)
        # print("[Info] batch: {}".format(batch))
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
        # lab = torch.cat([_['lab'] for _ in batch], dim=0).cuda()
        #im_name = torch.cat([_['im_name'] for _ in batch], dim=0)
        input = { 'img': img, 'msk': msk}
        # print("[Info] input: {}".format(input))

        losses = process_batch(input, 'train')

        if step % 10 == 0:
            print('\r{0} - '.format(step) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
        if step % 100 == 0:
            print('\n', end='')
        [write_tfboard(losses[_], epoch * config.stepsPerEpoch + step, _) for _ in losses]
        step = step + 1

    model.save(epoch+1, optimizer, savePath)

if __name__ == "__main__":
    # Setting
    dataConfig = config.modelConfig[config.backbone]

    # Random
    seed = 1
    torch.backends.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Building dataset
    print("[Info] Build dataset")
    trainData = Dataset('train', config.batch_size, dataConfig['img_size'], dataConfig['map_size'], dataConfig['norms'], seed)
    evalData = Dataset('eval', config.batch_size, dataConfig['img_size'], dataConfig['map_size'], dataConfig['norms'], seed)
    print("[Info] Built dataset\n\n")

    # Saving model
    modelName = '{0}_{1}'.format(config.backbone, config.mapType)
    savePath =config.saveDir + modelName + '/'

    # Building model
    model = Model(load_pretrain = False)
    optimizer = optim.Adam(model.model.parameters(), lr=config.learningRate, weight_decay=config.weightDecay)
    model.model.cuda()
    lossCSEFunc = nn.CrossEntropyLoss().cuda()
    lossL1Func = nn.L1Loss().cuda()
    maxPool = nn.MaxPool2d(19).cuda()
    summary_writer = SummaryWriter(savePath + 'logs/')

    # print(model.model)

    lastEpoch = 0
    for epoch in range(lastEpoch, config.maxEpochs):
        run_epoch(epoch)
        pass
