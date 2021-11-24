# from imutils import paths
# imglist = list(paths.list_images('/home/sean/faceforensic/train/real'))
# print(imglist)

# from convLSTM import ConvLSTMCell
# rnn = ConvLSTMCell(728, 1, (3, 3), True)
# print(rnn)

import torch
from vit_map import Model

model = Model()


img = torch.randn(6, 3, 299, 299)
preds = model.model(img)
print(img.shape)
print(preds)