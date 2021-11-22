import torch.nn as nn
import torch
from convLSTM import ConvLSTMCell


from imutils import paths

imglist = list(paths.list_images('/home/sean/faceforensic/train/real'))
print(imglist)


rnn = ConvLSTMCell(728, 1, (3, 3), True)
# print(rnn)
