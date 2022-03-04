import numpy as np
import os, shutil
import random
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn

from focalLoss import focal_loss
from config import Config as config
from dataset import Dataset
from templates import get_templates
from scipy.io import savemat

if config.backbone == 'xcp':
  # from xception import Model
  from xception_multiframe import Model
elif config.backbone == 'vgg':
  from vgg import Model


def write_tfboard(item, itr, name):
    summary_writer.add_scalar('{0}'.format(name), item, itr)

def calculate_losses_train(batch):
    img = batch['img']
    msk = batch['msk']
    lab = batch['lab']
    x, mask = model.model(img)
    loss_l1 = lossL1Func(mask, msk)
    loss_cse = lossCSEFunc(x, lab)
    loss = loss_l1 + loss_cse
    pred = torch.max(x, dim=1)[1]
    acc = (pred == lab).float().mean()
    return { 'loss': loss, 'loss_l1': loss_l1, 'loss_cse': loss_cse, 'acc': acc }

def process_batch_train(batch, mode):
    if mode == 'train':
        model.model.train()
        losses = calculate_losses_train(batch)
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
    elif mode == 'eval':
        model.model.eval()
        with torch.no_grad():
            losses = calculate_losses_train(batch)
    return losses

def run_epoch(epoch):
    print("[Info] Epoch: {}/{}".format(epoch, config.maxEpochs))
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
        lab = torch.cat([_['lab'] for _ in batch], dim=0).cuda()
        #im_name = torch.cat([_['im_name'] for _ in batch], dim=0)
        input = { 'img': img, 'msk': msk, 'lab': lab }
        # print("[Info] input: {}".format(input))

        losses = process_batch_train(input, 'train')

        if step % 10 == 0:
            print('\r{0} - '.format(step) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
        if step % 100 == 0:
            print('\n', end='')
        [write_tfboard(losses[_], epoch * config.stepsPerEpoch + step, _) for _ in losses]
        step = step + 1

    model.save(epoch+1, optimizer, savePath)


def calculate_losses_test(batch):
  img = batch['img']
  msk = batch['msk']
  lab = batch['lab']
  # x, mask, vec = MODEL.model(img)
  x, mask = model.model(img)
  loss_l1 = lossL1Func(mask, msk)
  loss_cse = lossCSEFunc(x, lab)
  loss = loss_l1 + loss_cse
  pred = torch.max(x, dim=1)[1]
  acc = (pred == lab).float().mean()
  res = { 'lab': lab, 'msk': msk, 'score': x, 'pred': pred, 'mask': mask }
  results = {}
  for r in res:
    results[r] = res[r].squeeze().cpu().numpy()
  return { 'loss': loss, 'loss_l1': loss_l1, 'loss_cse': loss_cse, 'acc': acc }, results

def process_batch_test(batch, mode):
  model.model.eval()
  with torch.no_grad():
    losses, results = calculate_losses_test(batch)
  return losses, results

def testing():
    for e in range(0, config.maxEpochs, 1):
        resultdir = '{0}results/{1}/'.format(config.saveDir, e)
        if os.path.exists(resultdir):
            shutil.rmtree(resultdir)
        os.makedirs(resultdir, exist_ok=True)
        # MODEL.load(e, MODEL_DIR)
        # testData = get_dataset()

        step = 0
        realTestLoader = testData.datasets[0].loader
        fakeTestLoader = testData.datasets[1].loader

        # Real
        for batch in realTestLoader:
            img = batch['img'].cuda()
            msk = batch['msk'].cuda()
            lab = batch['lab'].cuda()
            #im_name = torch.cat([_['im_name'] for _ in batch], dim=0)
            input = { 'img': img, 'msk': msk, 'lab': lab }
            # print(input)

            step = step + 1
            losses, results = process_batch_test(input, 'test')
            savemat('{0}{1}_{2}.mat'.format(resultdir, 0, step), results)

            if step % 10 == 0:
                print('{0} - '.format(step) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]))

        # Fake
        step = 0
        for batch in fakeTestLoader:
            img = batch['img'].cuda()
            msk = batch['msk'].cuda()
            lab = batch['lab'].cuda()
            #im_name = torch.cat([_['im_name'] for _ in batch], dim=0)
            input = { 'img': img, 'msk': msk, 'lab': lab }
            # print(input)

            step = step + 1
            losses, results = process_batch_test(input, 'test')
            savemat('{0}{1}_{2}.mat'.format(resultdir, 1, step), results)

            if step % 10 == 0:
                print('{0} - '.format(step) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]))
        
            print()

    print('Testing complete')



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
    testData = Dataset('test', config.batch_size, dataConfig['img_size'], dataConfig['map_size'], dataConfig['norms'], seed)
    
    print("[Info] Built dataset\n\n")

    # Saving model
    modelName = '{0}_{1}'.format(config.backbone, config.mapType)
    savePath = config.saveDir + modelName + '/'

    # Building model
    templates = None
    if config.mapType in ['tmp', 'pca_tmp']:
        templates = get_templates()
    model = Model(config.mapType, templates, 2, False)
    optimizer = optim.Adam(model.model.parameters(), lr=config.learningRate, weight_decay=config.weightDecay)
    model.model.cuda()
    
    # lossCSEFunc = nn.CrossEntropyLoss().cuda()
    lossCSEFunc = focal_loss().cuda()
    lossL1Func = nn.L1Loss().cuda()
    maxPool = nn.MaxPool2d(19).cuda()
    summary_writer = SummaryWriter(savePath + 'logs/')

    # print(model.model)

    lastEpoch = 0
    for epoch in range(lastEpoch, config.maxEpochs):
        # Training
        run_epoch(epoch)

        # Testing
        testing()

        # pass

