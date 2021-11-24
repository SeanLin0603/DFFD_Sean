import os
import torch
import torch.nn as nn
from vit_pytorch import ViT

from convLSTM import ConvLSTMCell

class VitLstm(nn.Module):
    def  __init__(self):
        super().__init__()
        self.map = ConvLSTMCell(1, 1, (3, 3), True)

        self.vit = ViT(
            image_size = 299,
            patch_size = 13,
            num_classes = 1000,
            dim = 361,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    def forward(self, input):
        x = self.vit(input)
        batchSize = x.shape[0]
        x = torch.reshape(x, (batchSize, 1 , 19, 19))
        
        # random initialization
        h0 = torch.rand(1, 1, 19, 19).cuda()
        c0 = torch.rand(1, 1, 19, 19).cuda()

        frame = torch.split(x, 1)
        # frame.shape: [1, 1, 19, 19]
        # print("[Info] frame shape: {}".format(len(frame)))

        numFrame = len(frame)
        hList = []
        cList = []

        for i in range(numFrame):
            if i == 0:
                h, c = self.map(frame[i], [h0, c0])
            else:
                h, c = self.map(frame[i], [hList[i - 1], cList[i - 1]])
            hList.append(h)
            cList.append(c)
        
        hTuple = tuple(hList)
        mask = torch.cat(hTuple, dim=0)
        # mask.shape: [6, 1, 19, 19]
        # print("[Info] mask shape: {}".format(mask.shape))

        return mask
        

class Model:
    def __init__(self):
        model = VitLstm()
        self.model = model

    def load(self, epoch, model_dir):
        filename = '{0}{1:06d}.tar'.format(model_dir, epoch)
        print('Loading model from {0}'.format(filename))
        if os.path.exists(filename):
            state = torch.load(filename)
            self.model.load_state_dict(state['net'])
        else:
            print('Failed to load model from {0}'.format(filename))

    def save(self, epoch, optim, model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        state = {'net': self.model.state_dict(), 'optim': optim.state_dict()}
        torch.save(state, '{0}/{1:06d}.tar'.format(model_dir, epoch))
        print('Saved model `{0}`'.format(epoch))

